import time
import sys
import win32com.client as win32

def message(msg, sleep=1):
    for l in msg:
        sys.stdout.write(l)
        sys.stdout.flush()
        time.sleep(0.025)
    time.sleep(sleep)

def send_code(subject, recipient, body):
    outlook = win32.Dispatch('outlook.application')
    mail = outlook.CreateItem(0)
    mail.To = recipient
    mail.Subject = subject
    mail.Body = body
    mail.Send()

def print_message_ws(m):
    words = m.split()
    for i in range(len(words)):
        if i >= 13 and (i - 13) % 14 == 0:
            words[i] += '\n'
        else:
            words[i] += ' '
    words.append("\n\n")
    return ''.join(words)