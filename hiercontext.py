import re


code = """
def total_seconds(td):
    \"\"\"For those with older versions of Python, a pure-Python
    implementation of Python 2.7's :meth:`~datetime.timedelta.total_seconds`.

    Args:
        td (datetime.timedelta): The timedelta to convert to seconds.
    Returns:
        float: total number of seconds

    >>> td = timedelta(days=4, seconds=33)
    >>> total_seconds(td)
    345633.0
    \"\"\"
    a_milli = 1000000.0
    td_ds = td.seconds + (td.days * 86400)  # 24 * 60 * 60
    td_micro = td.microseconds + (td_ds * a_milli)
    return td_micro / a_milli
"""
code = re.sub(r'"""(.*?)"""', '', code, flags=re.DOTALL)

pass