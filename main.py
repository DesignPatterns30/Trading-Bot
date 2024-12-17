import sys
import json
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QMessageBox,
    QLabel,
    QProgressBar,
    QDialog,
    QSlider,
    QInputDialog,
    QTextEdit,
    QListWidget,
    QSizePolicy,
    QSpacerItem,
)
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QFont
from SMTP import SMTPClient
from Observer import PrintObserver, MailObserver
from TradingBot import TradingBot
import os
from dotenv import load_dotenv
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import pyqtSignal


load_dotenv()
USERNAME = os.getenv("USER_NAME")
PASSWORD = os.getenv("PASSWORD")
SERVER = "smtp.gmail.com"
PORT = 587
OBSERVERS_FILE = "observers.json"  # File to store cached observers
# Above are the setup to use the Google's SMTP server,


class BotThread(
    QThread
):  # This is the bot thread where non-ui related operations are performed
    def __init__(self, bot):
        super().__init__()
        self.bot = bot

    def run(self):
        self.bot.run()

    def stop(self):
        self.bot.stop()
        self.quit()


class ProgressDialog(
    QDialog
):  # This is the dialog that shows the progress of the bot, when we run the program
    bot_stopped = pyqtSignal()  # Signal to notify when the bot is stopped

    def __init__(self, sleep_time, bot_thread):
        super().__init__()
        self.setWindowTitle("Progress")
        self.resize(300, 150)  # Setting the size of the dialog
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)  # Margin and spacing
        layout.setSpacing(5)

        spacer_top = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer_top)

        self.status_label = QLabel("Waiting to fetch data...", self)
        self.status_label.setAlignment(Qt.AlignCenter)  # Labeling and its alignment
        layout.addWidget(self.status_label)

        self.spinner_label = QLabel(self)
        self.spinner_movie = QMovie("spinner.gif")  # Path to your spinner GIF
        self.spinner_label.setMovie(self.spinner_movie)
        self.spinner_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.spinner_label)
        self.spinner_movie.start()

        spacer_middle1 = QSpacerItem(
            20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding
        )  # Spacer
        layout.addItem(spacer_middle1)

        self.progress_bar = QProgressBar(self)  # Progress bar , while fetching data
        layout.addWidget(self.progress_bar)

        spacer_middle2 = QSpacerItem(
            20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding
        )  # Spacer
        layout.addItem(spacer_middle2)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(
            self.stop_bot
        )  # Stop button and its onClick action
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch()  # Layout
        layout.addLayout(button_layout)

        spacer_bottom = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        layout.addItem(spacer_bottom)  # Spacer

        self.setLayout(layout)
        self.bot_thread = bot_thread  # Bot's thread

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def set_status_label(self, text):
        self.status_label.setText(text)
        if "Fetched data successfully!" in text:
            self.spinner_label.hide()
        else:
            self.spinner_label.show()
            self.spinner_movie.start()

    def stop_bot(self):
        self.bot_thread.stop()
        self.bot_stopped.emit()  # Emit the signal when the bot is stopped
        self.close()  # Logic to actively reflect data to UI and to stop the bot


class App(QWidget):  # This is the main UI segment of our program
    def __init__(self):
        super().__init__()
        self.sleep_time = 25
        self.initUI()
        self.bot = TradingBot(self.sleep_time)
        self.observer = PrintObserver()
        self.bot.register_observer(self.observer)
        self.load_observers()

    def initUI(self):  # Initial ui initiation
        self.setWindowTitle("Trading Bot")
        self.resize(400, 500)
        layout = QVBoxLayout()
        self.title = QLabel("Trading Bot", self)
        font = QFont()
        font.setPointSize(20)
        self.title.setFont(font)
        self.title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title)

        self.run_button = QPushButton("Run Program", self)
        button_font = QFont()
        button_font.setPointSize(13)
        self.run_button.setMinimumHeight(50)
        self.run_button.setFont(button_font)
        self.run_button.clicked.connect(self.run_program)
        layout.addWidget(self.run_button)

        self.add_observer_button = QPushButton("Add/Remove Observer", self)
        self.add_observer_button.setMinimumHeight(50)
        self.add_observer_button.setFont(button_font)
        self.add_observer_button.clicked.connect(self.manage_observers)
        layout.addWidget(self.add_observer_button)

        self.exit_button = QPushButton("Exit", self)
        self.exit_button.setMinimumHeight(50)
        self.exit_button.setFont(button_font)
        self.exit_button.clicked.connect(self.exit_program)
        layout.addWidget(self.exit_button)

        label_font = QFont()
        label_font.setPointSize(12)
        self.slider_label = QLabel(f"Sleep Time: {self.sleep_time} seconds", self)
        self.slider_label.setFont(label_font)
        self.slider_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.slider_label)
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(1)
        self.slider.setMaximum(500)
        self.slider.setValue(self.sleep_time)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(5)
        self.slider.valueChanged.connect(self.update_sleep_time)
        layout.addWidget(self.slider)
        self.log_output = QTextEdit(self)
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)
        self.log_output.hide()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        self.setLayout(layout)

    def update_sleep_time(self, value):  # Method to update sleep time
        self.sleep_time = value
        self.bot.sleep_time = value
        self.slider_label.setText(f"Sleep Time: {value} seconds")

    def run_program(self):  # Method to run the program
        self.log_output.show()  # Show the log output
        self.bot = TradingBot(self.sleep_time)  # Create a new bot
        self.observer = PrintObserver()  # Create a new observer
        self.bot.register_observer(self.observer)  # Register the observer
        self.load_observers()  # Load the observers
        self.bot_thread = BotThread(self.bot)
        self.progress_dialog = ProgressDialog(self.sleep_time, self.bot_thread)
        self.progress_dialog.bot_stopped.connect(self.clear_logs)  # Connect the signal
        self.bot.update_progress.connect(self.progress_dialog.update_progress)
        self.bot.fetched_data.connect(
            lambda: self.progress_dialog.set_status_label("Fetched data successfully!")
        )
        self.bot.update_log.connect(self.update_log_output)
        self.progress_dialog.show()
        self.progress_dialog.set_status_label("Fetching data, please wait...")
        self.bot_thread.start()  # Run the bot thread

    def clear_logs(self):
        self.log_output.clear()

    def update_log_output(self, log_message):  # Method to update log output
        self.log_output.append(log_message)

    def manage_observers(self):  # Manage observers ui, where we add or remove observers
        dialog = QDialog(self)
        dialog.setWindowTitle("Manage Observers")
        dialog.resize(300, 400)
        layout = QVBoxLayout()
        observer_list = QListWidget(dialog)
        observers = self.load_cached_observers()
        for observer in observers:
            observer_list.addItem(f"{observer['name']} ({observer['email']})")
        layout.addWidget(observer_list)
        add_button = QPushButton("Add Observer", dialog)
        add_button.clicked.connect(lambda: self.add_observer(observer_list))
        layout.addWidget(add_button)
        remove_button = QPushButton("Remove Observer", dialog)
        remove_button.clicked.connect(lambda: self.remove_observer(observer_list))
        layout.addWidget(remove_button)
        dialog.setLayout(layout)
        dialog.exec_()

    def add_observer(self, observer_list):  # Method to add observer
        name, ok1 = QInputDialog.getText(self, "Observer Name", "Enter observer name:")
        if ok1:
            email, ok2 = QInputDialog.getText(
                self, "Observer Email", "Enter observer email:"
            )
            if ok2:
                smtp_client = SMTPClient(SERVER, PORT, USERNAME, PASSWORD)
                mail_observer = MailObserver(smtp_client, USERNAME, email)
                self.bot.register_observer(
                    mail_observer
                )  # Bot's register observer is called
                self.cache_observer(name, email)
                observer_list.addItem(f"{name} ({email})")
                QMessageBox.information(
                    self, "Info", f"Observer {name} added with email {email}"
                )

    def remove_observer(self, observer_list):  # Remove observers
        selected_items = observer_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "No observer selected.")
            return
        for item in selected_items:
            observer_text = item.text()
            name, email = observer_text.split(" (")
            email = email.rstrip(")")
            self.remove_cached_observer(name, email)  # Remove cached observer
            observer_list.takeItem(observer_list.row(item))
            QMessageBox.information(self, "Info", f"Observer {name} removed.")

    def cache_observer(self, name, email):  # Cache observer into json file
        observers = self.load_cached_observers()
        observers.append({"name": name, "email": email})
        with open(OBSERVERS_FILE, "w") as f:
            json.dump(observers, f)

    def remove_cached_observer(
        self, name, email
    ):  # Remove cached observer from json file
        observers = self.load_cached_observers()
        observers = [
            obs for obs in observers if obs["name"] != name or obs["email"] != email
        ]
        with open(OBSERVERS_FILE, "w") as f:
            json.dump(observers, f)

    def load_cached_observers(self):  # Load cached observers from json file
        try:
            if os.path.exists(OBSERVERS_FILE):
                with open(OBSERVERS_FILE, "r") as f:
                    return json.load(f)
            else:
                return []
        except:
            return []

    def load_observers(self):  # Load observers from json file, and register them
        observers = self.load_cached_observers()
        for observer in observers:
            smtp_client = SMTPClient(SERVER, PORT, USERNAME, PASSWORD)
            mail_observer = MailObserver(smtp_client, USERNAME, observer["email"])
            self.bot.register_observer(mail_observer)

    def exit_program(self):  # Method to exit the program
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
