#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
from itertools import chain
from numpy import float64


# In[2]:


def month_str_to_num(month: str):
    return {
        "Jan": 0,
        "Feb": 1,
        "Mar": 2,
        "Apr": 3,
        "May": 4,
        "Jun": 5,
        "Jul": 6,
        "Aug": 7,
        "Sep": 8,
        "Oct": 9,
        "Nov": 10,
        "Dec": 11
    }[month]

def days_in_past_months(month: int, _is_leap_year: bool):
    return sum([31, 28 if not _is_leap_year else 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][:month])

def is_leap_year(year: int):
    true_year = year + 1986
    return true_year in [1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020]

def num_past_leap_years(year):
    true_year = year + 1986
    return len(list(filter(lambda list_year: list_year < true_year, [1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020])))

def days_in_past_years(year: int):
    past_leap_years = num_past_leap_years(year)
    return past_leap_years*366 + (year - past_leap_years)*365

def date_to_days(date: str):
    str_day, str_month, str_year = date.split("-")
    day = int(str_day)
    month = month_str_to_num(str_month)
    year = int(str_year) + (0 if int(str_year) >= 86 else 100) - 86
    
    return day         + days_in_past_months(month, is_leap_year(year))         + days_in_past_years(year)


# In[3]:


def get_subsequences(data, sequences_length: int):
    subsequences = [data[start_date:start_date + sequences_length] for start_date in range(len(data) - sequences_length)]
    return torch.stack([torch.tensor(arr) for arr in subsequences])


# In[32]:


def interpolate_missing_dates(data: pd.DataFrame):
    missing_date_indices = [(idx, date - data["date"][idx-1] - 1) for idx, date in enumerate(data["date"]) if idx != 0 and date-1 != data["date"][idx-1]]
    fixed_indices = list(range(len(data)))
    for idx, num_missing_dates in reversed(missing_date_indices):
        fixed_indices[idx:idx] = [float64("nan")] * num_missing_dates
    return data.reindex(fixed_indices).interpolate().reset_index(drop=True)


# In[52]:


def load_data(csv_file, sequences_length=50):
    pd_data = pd.read_csv(csv_file)
    pd_data.date = pd_data.date.apply(date_to_days)
    pd_data = interpolate_missing_dates(pd_data)
    date_sequences = get_subsequences(pd_data.values, sequences_length)
    target_prices = torch.tensor([date_sequences[idx + 1][-1][1] for idx in range(len(date_sequences) - 1)])
    date_sequences = date_sequences[:-1]
    dataset = torch.utils.data.TensorDataset(date_sequences, target_prices)
    train_data, test_data = torch.utils.data.random_split(
        dataset=dataset, 
        lengths=(len(dataset) - len(dataset)//10, len(dataset)//10), 
        generator=torch.Generator().manual_seed(0)
    )
    return train_data, test_data

