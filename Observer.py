from interface import implements, Interface


class Observer(Interface):  # Our observer interface
    def update(self, signal):
        raise NotImplementedError("Subclass must implement abstract method")


class PrintObserver(implements(Observer)):  # Concrete observer class, console observer
    def update(self, signal):
        print(f"Received signal: {signal}")


class MailObserver(implements(Observer)):  # Mail observer concrete class,
    def __init__(self, client, username, mail):
        self.username = username
        self.mail = mail
        self.client = client  # Each MailObserver object will have a reference to the SMTPClient object

    def update(
        self, signal
    ):  # This method works as if we pull notifications from the bot, we make the SMTP client send us an email
        subject = "Trading Bot Signal"
        text = f"The signal received is: {signal}\n"
        self.client.add_recipient(self.mail)
        self.client.send_email(subject, text)
