'''this is an test'''

import datetime


def start_date():
    '''Get today date'''
    if datetime.datetime.now().weekday() < 6:
        nextday_date = (datetime.date.today() + datetime.timedelta(days=1))
        nextday = datetime.datetime(nextday_date.year,
                                    nextday_date.month, nextday_date.day, 13, 0, 0, 0)
    else:
        nextday_date = (datetime.date.today() + datetime.timedelta(days=2))
        nextday = datetime.datetime(nextday_date.year,
                                    nextday_date.month, nextday_date.day, 13, 0, 0, 0)
    start_date = nextday.timestamp()
    return start_date
# print(nextday)
# print(start_date())
# utc_dt = datetime.datetime.utcfromtimestamp(start_date())
# print(utc_dt)
