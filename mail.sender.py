import smtplib
from email.message import EmailMessage
import ssl

def send_email(image_path, person_name):
    sender = "ht8734519@gmail.com"
    password = "lzok owtm zabk ggkb"
    receiver = "lamn64265@gmail.com"

    msg = EmailMessage()
    msg["Subject"] = "Thông báo phát hiện người"
    msg["From"] = sender
    msg["To"] = receiver

    msg.set_content(f"Hệ thống đã phát hiện: {person_name}, Đây là email được gửi tự động.")

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender, password)
        server.send_message(msg)

    print("Đã gửi email thông báo!")

def open_browser():
    import webbrowser
    url = "https://youtu.be/BCvWHPbmNxc"
    webbrowser.open(url)