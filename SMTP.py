import smtplib
import logging


##This is our SMTP client class, which is responsible for sending emails
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
# Setting up the logger


class SMTPClient:  # This is our SMTPClient class, which is used by our bot to send emails
    def __init__(self, server, port, username, password):
        self.server = server
        self.port = port
        self.username = username
        self.password = password
        self.recipients = []

    def add_recipient(self, recipient):  # Method to add recipients
        self.recipients.append(recipient)

    def send_email(self, subject, text):  # Method to send email
        if not self.recipients:
            logger.error("No recipients added.")
            return

        message = f"""\
From: {self.username}
To: {", ".join(self.recipients)}
Subject: {subject}

{text}
"""
        try:
            server = smtplib.SMTP(self.server, self.port)  # Using smtplib to send email
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(self.username, self.password)
            server.sendmail(self.username, self.recipients, message)
            server.quit()
        except smtplib.SMTPAuthenticationError as e:
            pass
        except Exception as e:
            pass
