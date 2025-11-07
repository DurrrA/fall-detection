from email_provider import EmailProvider
from sms_provider import SMSProvider

class Notifier:
    def __init__(self, email_config, sms_config):
        self.email_provider = EmailProvider(email_config)
        self.sms_provider = SMSProvider(sms_config)

    def send_fall_alert(self, message):
        self.send_email_alert(message)
        self.send_sms_alert(message)

    def send_email_alert(self, message):
        subject = "Fall Detection Alert"
        self.email_provider.send_email(subject, message)

    def send_sms_alert(self, message):
        self.sms_provider.send_sms(message)