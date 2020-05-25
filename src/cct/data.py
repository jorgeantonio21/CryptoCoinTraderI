import asyncio
import copra
from copra.rest import Client

import numpy as np
import pandas as pd

import time
import datetime


class DataClient():
    """
    Getting data for training
    """
    def __init__(self):
        self.loop = asyncio.get_event_loop()
        self.client = Client(self.loop)
        self.max_dataframe_size = 350 # this is the length of a DataFrame given by historic_rates method for Client

    async def get_stats_hist(self, product_id, begin, end, granularity=300):
        """
        Using UTC time
        Granularity should be one of the values 60, 300, 900, 3600, 21600, 86400
        begin and end correspond to the starting time and the end time, assuming these are represented in UTC time hour, allowed None values
        """

        if (end > datetime.datetime.utcnow()):
            raise ValueError('End time cannot be set in the future')
        
        if (begin == None | end == None):
            stats = await self.client.historic_rates(product_id, granularity)
            hist_stats = {}
            for i in range(len(stats)):
                    hist_stats[stats[i][0]] = stats[i][1:]

            df = pd.DataFrame(hist_stats).transpose()
            df.columns = ['open', 'high', 'low', 'volume', 'last'] 

            return df
        dt = datetime.timedelta(hours=(granularity * self.max_dataframe_size)) 
        current_time = begin
        hist_stats = {}

        while (current_time < end):

            if (current_time + dt < end):
                #Instantiate historic_rates on the period (current_time, current_time + dt)
                stats = await self.client.historic_rates(product_id, granularity, start=current_time,
                                                            stop=current_time + dt)

                for i in range(len(stats)):
                    hist_stats[stats[i][0]] = stats[i][1:]
                
                current_time += dt
            elif (current_time + dt >= end):
                #Instantiate historic_rates instead on the period (current_time, end)
                stats = await self.client.historic_rates(product_id, granularity, start=current_time, 
                                                            stop=end)

                for i in range(len(stats)):
                    hist_stats[stats[i][0]] = stats[i][1:]

                current_time = end    

        df = pd.DataFrame(hist_stats).transpose()
        df.columns = ['open', 'high', 'low', 'volume', 'last']             
        return df

    def close_section(self):
        self.loop.run_until_complete(self.client.close())




now = datetime.datetime.now()
data = DataClient()
print(data.get_stats_hist('ETH-EUR', now - datetime.timedelta(seconds=10), now, granularity=60))
data.close_section()
        
