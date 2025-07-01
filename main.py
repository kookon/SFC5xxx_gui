import sys
import csv
import time
import os
from datetime import datetime, timezone
from collections import deque
import numpy as np

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QCheckBox, QMessageBox, QStatusBar)
from PyQt6.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont
import pyqtgraph as pg

# --- Sensirion Driver Imports ---
from sensirion_shdlc_driver import ShdlcSerialPort, ShdlcConnection
from sensirion_shdlc_sfc5xxx import (Sfc5xxxShdlcDevice, Sfc5xxxScaling,
                                     Sfc5xxxMediumUnit, Sfc5xxxUnitPrefix,
                                     Sfc5xxxUnit, Sfc5xxxUnitTimeBase)

# --- Configuration ---
COM_PORT = 'COM3'
MAX_DATA_POINTS = 600
POLLING_INTERVAL_MS = 100
RECONNECTION_INTERVAL_MS = 5000
AVERAGING_WINDOW = 10


# --- Custom Time Axis for Graph ---
class TimeAxisItem(pg.AxisItem):
    def tickStrings(self, values, scale, spacing):
        return [datetime.fromtimestamp(value, tz=timezone.utc).strftime('%H:%M:%S') for value in values]


# --- Data Acquisition Thread ---
class DataThread(QThread):
    newData = pyqtSignal(float, float, int)
    disconnected = pyqtSignal()

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.running = True

    def run(self):
        while self.running:
            try:
                status_tuple = self.device.read_device_status()
                status_code = status_tuple[0]

                measured_value = self.device.read_measured_value(Sfc5xxxScaling.USER_DEFINED)
                temperature = self.device.measure_temperature()
                self.newData.emit(measured_value, temperature, status_code)
            except Exception as e:
                print(f"Device disconnected: {e}")
                self.disconnected.emit()
                self.running = False
                break
            self.msleep(POLLING_INTERVAL_MS)

    def stop(self):
        self.running = False
        self.wait()


# --- Main Window ---
class MainWindow(QMainWindow):
    ERROR_CODES = {
        1: "Boot Error", 2: "Cmd Post Processing Error", 4: "Input Supply out of Range",
        8: "Valve Supply out of Range", 16: "Signal Processor Init", 32: "Sensor Comm Error",
        64: "Setpoint Input Error", 128: "Actuator Output Error", 256: "Signal Output Error",
        512: "Signal Buffer Error", 1024: "Missing Gas Pressure"
    }

    STATUS_EMOJI = {
        "ok": "😊", "error": "🙁", "disconnected": "⚪"
    }
#💩
    def __init__(self, device, initial_setpoint):
        super().__init__()
        self.device = device
        self.current_setpoint = initial_setpoint
        self.setWindowTitle("SFC5500 Mass Flow Controller GUI")

        self.time_data = deque(maxlen=MAX_DATA_POINTS)
        self.setpoint_data = deque(maxlen=MAX_DATA_POINTS)
        self.measured_data = deque(maxlen=MAX_DATA_POINTS)

        self.setup_ui()

        # --- Connections & Timers ---
        self.setpoint_button.clicked.connect(self.set_flow_setpoint)
        self.setpoint_input.returnPressed.connect(self.set_flow_setpoint)
        self.averaging_checkbox.stateChanged.connect(self.update_graph)
        self.reconnect_timer = QTimer(self)
        self.reconnect_timer.timeout.connect(self.attempt_reconnection)

        self.setpoint_input.setText(f"{self.current_setpoint:.2f}")
        self.setpoint_indicator_label.setText(f"Setpoint: {self.current_setpoint:.2f} sccm")
        self.start_data_thread()

    def setup_ui(self):
        self.setStatusBar(QStatusBar(self))
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        time_axis = TimeAxisItem(orientation='bottom')
        self.graphWidget = pg.PlotWidget(axisItems={'bottom': time_axis})
        self.graphWidget.setBackground('w')
        self.graphWidget.setLabel('left', 'Flow (sccm)')
        self.graphWidget.setLabel('bottom', 'Time (UTC)')
        self.graphWidget.addLegend()
        self.graphWidget.showGrid(x=True, y=True)
        self.setpoint_line = self.graphWidget.plot(pen=pg.mkPen('r', width=2), name="Setpoint")
        self.measured_line = self.graphWidget.plot(pen=pg.mkPen('b', width=2), name="Measured")
        main_layout.addWidget(self.graphWidget, 3)

        controls_layout = QVBoxLayout()
        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)

        # Status Indicators (Temp, Setpoint)
        status_layout = QHBoxLayout()
        self.temp_label = QLabel("Temperature: -- °C")
        self.temp_label.setStyleSheet("font-size: 12pt;")
        status_layout.addWidget(self.temp_label)

        self.setpoint_indicator_label = QLabel("Setpoint: -- sccm")
        self.setpoint_indicator_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        status_layout.addWidget(self.setpoint_indicator_label)
        controls_layout.addLayout(status_layout)

        # Setpoint Input
        setpoint_layout = QHBoxLayout()
        self.setpoint_input = QLineEdit()
        self.setpoint_button = QPushButton("Set")
        setpoint_layout.addWidget(QLabel("Setpoint (sccm):"))
        setpoint_layout.addWidget(self.setpoint_input)
        setpoint_layout.addWidget(self.setpoint_button)
        controls_layout.addLayout(setpoint_layout)

        # Averaging Checkbox
        self.averaging_checkbox = QCheckBox(f"Enable Averaging ({AVERAGING_WINDOW}-point)")
        controls_layout.addWidget(self.averaging_checkbox)

        # Error status label
        self.error_label = QLabel("Status: OK")
        self.error_label.setStyleSheet("color: #555;")
        self.error_label.setWordWrap(True)
        controls_layout.addWidget(self.error_label)

        # This stretch pushes everything below it to the bottom
        controls_layout.addStretch()

        # Emoji Status Label (bottom-right)
        self.emoji_status_label = QLabel()
        font = QFont()
        font.setPointSize(48)
        self.emoji_status_label.setFont(font)
        controls_layout.addWidget(self.emoji_status_label, 0, Qt.AlignmentFlag.AlignRight)

        main_layout.addWidget(controls_widget, 1)

    def start_data_thread(self):
        self.data_thread = DataThread(self.device)
        self.data_thread.newData.connect(self.update_data)
        self.data_thread.disconnected.connect(self.handle_disconnection)
        self.data_thread.start()
        self.set_controls_enabled(True)
        self.set_status_emoji("ok")
        self.statusBar().showMessage("✅ Connected", 5000)

    def set_status_emoji(self, state):
        self.emoji_status_label.setText(self.STATUS_EMOJI.get(state, "❓"))

    def handle_disconnection(self):
        self.set_controls_enabled(False)
        self.set_status_emoji("disconnected")
        self.error_label.setText("Status: Disconnected")
        self.statusBar().showMessage("❌ Disconnected! Attempting to reconnect...")
        self.reconnect_timer.start(RECONNECTION_INTERVAL_MS)

    def attempt_reconnection(self):
        self.statusBar().showMessage("⏳ Attempting to reconnect...")
        try:
            port = ShdlcSerialPort(port=COM_PORT, baudrate=115200)
            self.device = Sfc5xxxShdlcDevice(ShdlcConnection(port), slave_address=0)
            self.device.get_serial_number()
            self.reconnect_timer.stop()
            self.set_user_unit()
            self.set_flow_setpoint(force=True)
            self.start_data_thread()
        except Exception as e:
            self.statusBar().showMessage("❌ Reconnect failed. Retrying...")

    def update_data(self, measured_value, temperature, status):
        if status == 0:
            self.set_status_emoji("ok")
        else:
            self.set_status_emoji("error")

        self.error_label.setText(self.parse_status(status))
        utc_timestamp = datetime.now(timezone.utc).timestamp()
        self.time_data.append(utc_timestamp)
        self.setpoint_data.append(self.current_setpoint)
        self.measured_data.append(measured_value)
        self.temp_label.setText(f"Temperature: {temperature:.2f} °C")
        self.setpoint_indicator_label.setText(f"Setpoint: {self.current_setpoint:.2f} sccm")
        self.log_data([utc_timestamp, self.current_setpoint, measured_value, temperature, status])
        self.update_graph()

    def parse_status(self, status):
        if status == 0:
            return "Status: OK"

        active_errors = [text for code, text in self.ERROR_CODES.items() if status & code]
        return f"Status: {', '.join(active_errors)} (Code: {status})"

    def get_log_filename(self):
        return f"{datetime.now().strftime('%Y-%m-%d')}_sfc5500_log.csv"

    def log_data(self, data_row):
        filename = self.get_log_filename()
        file_exists = os.path.isfile(filename)
        try:
            with open(f'log/{filename}', 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(
                        ['Unix Timestamp (UTC)', 'Setpoint (sccm)', 'Measured Value (sccm)', 'Temperature (C)',
                         'Device Status'])
                writer.writerow(data_row)
        except IOError as e:
            print(f"Could not write to log file: {e}")

    def update_graph(self):
        time_list = list(self.time_data)
        measured_list = list(self.measured_data)
        if self.averaging_checkbox.isChecked() and len(measured_list) >= AVERAGING_WINDOW:
            weights = np.ones(AVERAGING_WINDOW) / AVERAGING_WINDOW
            avg_measured = np.convolve(measured_list, weights, 'valid')
            self.measured_line.setData(time_list[AVERAGING_WINDOW - 1:], avg_measured)
        else:
            self.measured_line.setData(time_list, measured_list)
        self.setpoint_line.setData(time_list, list(self.setpoint_data))

    def closeEvent(self, event):
        self.reconnect_timer.stop()
        if hasattr(self, 'data_thread'): self.data_thread.stop()
        try:
            self.device.set_setpoint(0, Sfc5xxxScaling.USER_DEFINED)
            self.device.device_reset()
        except Exception as e:
            print(f"Could not reset device on close: {e}")
        event.accept()

    def set_controls_enabled(self, enabled):
        self.setpoint_input.setEnabled(enabled)
        self.setpoint_button.setEnabled(enabled)

    def set_user_unit(self):
        unit = Sfc5xxxMediumUnit(Sfc5xxxUnitPrefix.MILLI, Sfc5xxxUnit.STANDARD_LITER, Sfc5xxxUnitTimeBase.MINUTE)
        self.device.set_user_defined_medium_unit(unit)

    def set_flow_setpoint(self, force=False):
        try:
            setpoint = float(self.setpoint_input.text()) if not force else self.current_setpoint
            if 0 <= setpoint <= 50:
                self.device.set_setpoint(setpoint, Sfc5xxxScaling.USER_DEFINED)
                self.current_setpoint = setpoint
                self.setpoint_indicator_label.setText(f"Setpoint: {self.current_setpoint:.2f} sccm")
            else:
                QMessageBox.warning(self, "Warning", "Setpoint must be between 0 and 50 sccm.")
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to set setpoint: {e}")


def main():
    app = QApplication(sys.argv)
    try:
        port = ShdlcSerialPort(port=COM_PORT, baudrate=115200)
        device = Sfc5xxxShdlcDevice(ShdlcConnection(port), slave_address=0)
        # device.activate_calibration(0)
        unit = Sfc5xxxMediumUnit(Sfc5xxxUnitPrefix.MILLI, Sfc5xxxUnit.STANDARD_LITER, Sfc5xxxUnitTimeBase.MINUTE)
        device.set_user_defined_medium_unit(unit)
        initial_setpoint = device.get_setpoint(Sfc5xxxScaling.USER_DEFINED)
        print(f"Device connected. Initial setpoint: {initial_setpoint} sccm")
        window = MainWindow(device, initial_setpoint)
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        QMessageBox.critical(None, "Connection Error", f"Failed to connect to the device on {COM_PORT}.\n\nError: {e}")


if __name__ == '__main__':
    main()