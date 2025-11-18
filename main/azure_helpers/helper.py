import time
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from datetime import datetime
import pytz
import collections
import itertools

def clean_the_string(full_path):
    """
    Args:
        full_path (str): _description_

    Returns:
        instance_name (str): The GroupContainer Name
        file_info_dict (dict): 
            - filename      : my_document
            - file_extension: .pdf
            - full_path     : root_folder/abc/my_document.pdf
            - expire_time   : Saves Expiretime in Isoformat 
    """
    filename = Path(full_path).stem
    file_extention = Path(full_path).suffix
    # Remove the extension and filepaths
    # instance_name = re.sub(r'[^a-zA-Z0-9]', '', filename)
    instance_name = "".join(filter(str.isalnum, filename))
    instance_name = ("groupContainer"+instance_name).lower()
    stripped_filename = "".join(filter(str.isalnum, filename))
    file_info_dict = {
        "filename" : stripped_filename,
        "file_extension" : file_extention,
        "full_path" : full_path
    }
    return instance_name, file_info_dict

def get_time_now(mins_to_expire)->str:
    malaysia_timezone = pytz.timezone('Asia/Kuala_Lumpur')

    # Get the current time in the Malaysia timezone
    expire_time_malaysia = (datetime.now(malaysia_timezone)+ timedelta(minutes=mins_to_expire)).isoformat()# .replace(tzinfo=None).isoformat("#", "seconds")
    return expire_time_malaysia

def process_expired(starttime):
    # Get Current Time
    malaysia_timezone = pytz.timezone('Asia/Kuala_Lumpur')
    current_time = (datetime.now(malaysia_timezone)).replace(tzinfo=None)
    expire_time = datetime.fromisoformat(starttime) 
    if expire_time > current_time:
        return True
    else:
        return False

def time_human_readable(iso_string):
    # Convert ISO string to datetime object
    dt_object = datetime.fromisoformat(iso_string)

    # Get today's date and yesterday's date for comparison
    today = date.today()
    yesterday = today - timedelta(days=1)

    # Determine if the date is today or yesterday
    if dt_object.date() == today:
        date_part = "Today"
    elif dt_object.date() == yesterday:
        date_part = "Yesterday"
    else:
        # If not today or yesterday, format as a standard date
        date_part = dt_object.strftime("%Y-%m-%d")

    # Format the time part with AM/PM
    time_part = dt_object.strftime("%I:%M %p")

    return f"{date_part}, {time_part}"

def count_iter_items(iterable):
    counter = itertools.count()
    collections.deque(zip(iterable, counter), maxlen=0)
    return next(counter)