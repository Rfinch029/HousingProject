import requests
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
import openmeteo_requests
from datetime import datetime
from Preprocessing import prepData


IS_PREPROCESSING = False

def main():
    if (IS_PREPROCESSING):
        prepData()


if __name__ == '__main__':
    main()

