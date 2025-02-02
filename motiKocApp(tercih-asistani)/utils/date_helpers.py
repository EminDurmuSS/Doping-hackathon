from typing import Optional, List, Tuple, Dict, Union
from datetime import datetime, timedelta, date
import calendar

def parse_date(date_str: str) -> Optional[date]:
    """
    Parse a date string into a `date` object supporting multiple formats.

    Args:
        date_str (str): The date string to parse.

    Returns:
        Optional[date]: The parsed `date` object if successful; otherwise, `None`.
    """
    formats = [
        '%Y-%m-%d',
        '%d-%m-%Y',
        '%d.%m.%Y',
        '%d/%m/%Y'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None

def format_date(date_obj: Union[datetime, date], format_str: str = '%Y-%m-%d') -> str:
    """
    Format a `date` or `datetime` object into a string.

    Args:
        date_obj (Union[datetime, date]): The date or datetime object to format.
        format_str (str, optional): The format string. Defaults to '%Y-%m-%d'.

    Returns:
        str: The formatted date string.
    """
    return date_obj.strftime(format_str)

def get_date_range(start_date: date, end_date: date) -> List[date]:
    """
    Generate a list of dates between `start_date` and `end_date`, inclusive.

    Args:
        start_date (date): The starting date.
        end_date (date): The ending date.

    Returns:
        List[date]: A list of `date` objects from start to end date.
    
    Raises:
        ValueError: If `start_date` is after `end_date`.
    """
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date.")
    
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += timedelta(days=1)
    return date_list

def get_week_dates() -> Tuple[date, date]:
    """
    Get the start (Monday) and end (Sunday) dates of the current week.

    Returns:
        Tuple[date, date]: A tuple containing the Monday and Sunday dates of the current week.
    """
    today = datetime.now().date()
    monday = today - timedelta(days=today.weekday())
    sunday = monday + timedelta(days=6)
    return monday, sunday

def get_month_dates() -> Tuple[date, date]:
    """
    Get the start and end dates of the current month.

    Returns:
        Tuple[date, date]: A tuple containing the first and last dates of the current month.
    """
    today = datetime.now().date()
    start_date = today.replace(day=1)
    _, last_day = calendar.monthrange(today.year, today.month)
    end_date = today.replace(day=last_day)
    return start_date, end_date

def calculate_days_until(target_date: date) -> int:
    """
    Calculate the number of days from today until the `target_date`.

    Args:
        target_date (date): The target date.

    Returns:
        int: The number of days until the target date. Can be negative if the date is in the past.
    """
    today = datetime.now().date()
    delta = target_date - today
    return delta.days

def is_weekend(date_obj: date) -> bool:
    """
    Determine if a given date falls on a weekend.

    Args:
        date_obj (date): The date to check.

    Returns:
        bool: `True` if the date is Saturday or Sunday; otherwise, `False`.
    """
    return date_obj.weekday() >= 5  # 5 = Saturday, 6 = Sunday

def get_study_streak(dates: List[date]) -> int:
    """
    Calculate the number of consecutive study days in the provided list.

    Args:
        dates (List[date]): A list of study dates.

    Returns:
        int: The length of the longest consecutive study streak.
    """
    if not dates:
        return 0
    
    # Sort dates in descending order
    sorted_dates = sorted(dates, reverse=True)
    
    streak = 1
    for i in range(len(sorted_dates) - 1):
        if (sorted_dates[i] - sorted_dates[i + 1]).days == 1:
            streak += 1
        else:
            break
    
    return streak

def group_dates_by_month(dates: List[date]) -> Dict[str, List[date]]:
    """
    Group a list of dates by their respective months.

    Args:
        dates (List[date]): A list of dates to group.

    Returns:
        Dict[str, List[date]]: A dictionary where keys are month strings (e.g., '2024-04') 
                                and values are lists of dates within those months.
    """
    month_groups: Dict[str, List[date]] = {}
    
    for d in dates:
        month_key = d.strftime('%Y-%m')
        if month_key not in month_groups:
            month_groups[month_key] = []
        month_groups[month_key].append(d)
    
    return month_groups
