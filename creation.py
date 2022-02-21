# Necessary imports for this notebook
import os
from unicodedata import name
from neo4j import GraphDatabase

import numpy as np
import pandas as pd

import datetime
import time

import random

# For plotting
#%matplotlib inline

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid', {'axes.facecolor': '0.9'})


def generate_customer_profiles_table(n_customers, random_state=0):
    
    np.random.seed(random_state)
        
    customer_id_properties=[]
    
    # Generate customer properties from random distributions 
    for customer_id in range(n_customers):

        x_customer_id = np.random.uniform(0,100)
        y_customer_id = np.random.uniform(0,100)
        
        mean_amount = np.random.uniform(5,100) # Arbitrary (but sensible) value 
        std_amount = mean_amount/2 # Arbitrary (but sensible) value
        
        mean_nb_tx_per_day = np.random.uniform(0,4) # Arbitrary (but sensible) value 
        
        customer_id_properties.append([customer_id,
                                      x_customer_id, y_customer_id,
                                      mean_amount, std_amount,
                                      mean_nb_tx_per_day])
        
    customer_profiles_table = pd.DataFrame(customer_id_properties, columns=['CUSTOMER_ID',
                                                                      'x_customer_id', 'y_customer_id',
                                                                      'mean_amount', 'std_amount',
                                                                      'mean_nb_tx_per_day'])
    
    return customer_profiles_table

# n_customers = 5
# customer_profiles_table = generate_customer_profiles_table(n_customers, random_state = 0)
# customer_profiles_table.to_csv("customer_profiles",index=False)
# print(customer_profiles_table)

def generate_terminal_profiles_table(n_terminals, random_state=0):
    
    np.random.seed(random_state)
        
    terminal_id_properties=[]
    
    # Generate terminal properties from random distributions 
    for terminal_id in range(n_terminals):
        
        x_terminal_id = np.random.uniform(0,100)
        y_terminal_id = np.random.uniform(0,100)
        
        terminal_id_properties.append([terminal_id,
                                      x_terminal_id, y_terminal_id])
                                       
    terminal_profiles_table = pd.DataFrame(terminal_id_properties, columns=['TERMINAL_ID',
                                                                      'x_terminal_id', 'y_terminal_id'])
    
    return terminal_profiles_table

# n_terminals = 5
# terminal_profiles_table = generate_terminal_profiles_table(n_terminals, random_state = 0)
# print(terminal_profiles_table)


def get_list_terminals_within_radius(customer_profile, x_y_terminals, r):
    
    # Use numpy arrays in the following to speed up computations
    
    # Location (x,y) of customer as numpy array
    x_y_customer = customer_profile[['x_customer_id','y_customer_id']].values.astype(float)
    
    # Squared difference in coordinates between customer and terminal locations
    squared_diff_x_y = np.square(x_y_customer - x_y_terminals)
    
    # Sum along rows and compute suared root to get distance
    dist_x_y = np.sqrt(np.sum(squared_diff_x_y, axis=1))
    
    # Get the indices of terminals which are at a distance less than r
    available_terminals = list(np.where(dist_x_y<r)[0])
    
    # Return the list of terminal IDs
    return available_terminals



def generate_transactions_table(customer_profile, start_date = "2022-02-06", nb_days = 10):
    
    customer_transactions = []
    
    random.seed(int(customer_profile.CUSTOMER_ID))
    np.random.seed(int(customer_profile.CUSTOMER_ID))
    
    # For all days
    for day in range(nb_days):
        
        # Random number of transactions for that day 
        nb_tx = np.random.poisson(customer_profile.mean_nb_tx_per_day)
        
        # If nb_tx positive, let us generate transactions
        if nb_tx>0:
            
            for tx in range(nb_tx):
                
                # Time of transaction: Around noon, std 20000 seconds. This choice aims at simulating the fact that 
                # most transactions occur during the day.
                time_tx = int(np.random.normal(86400/2, 20000))
                
                # If transaction time between 0 and 86400, let us keep it, otherwise, let us discard it
                if (time_tx>0) and (time_tx<86400):
                    
                    # Amount is drawn from a normal distribution  
                    amount = np.random.normal(customer_profile.mean_amount, customer_profile.std_amount)
                    
                    # If amount negative, draw from a uniform distribution
                    if amount<0:
                        amount = np.random.uniform(0,customer_profile.mean_amount*2)
                    
                    amount=np.round(amount,decimals=2)
                    
                    if len(customer_profile.available_terminals)>0:
                        
                        terminal_id = random.choice(customer_profile.available_terminals)
                    
                        customer_transactions.append([time_tx+day*86400, day,
                                                      customer_profile.CUSTOMER_ID, 
                                                      terminal_id, amount])
            
    customer_transactions = pd.DataFrame(customer_transactions, columns=['TX_TIME_SECONDS', 'TX_TIME_DAYS', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT'])
    
    if len(customer_transactions)>0:
        customer_transactions['TX_DATETIME'] = pd.to_datetime(customer_transactions["TX_TIME_SECONDS"], unit='s', origin=start_date)
        customer_transactions=customer_transactions[['TX_DATETIME','CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT','TX_TIME_SECONDS', 'TX_TIME_DAYS']]
    
    return customer_transactions  
    


def add_frauds(customer_profiles_table, terminal_profiles_table, transactions_df):
    
    # By default, all transactions are genuine
    transactions_df['TX_FRAUD']=0
    transactions_df['TX_FRAUD_SCENARIO']=0
    
    # Scenario 1
    transactions_df.loc[transactions_df.TX_AMOUNT>220, 'TX_FRAUD']=1
    transactions_df.loc[transactions_df.TX_AMOUNT>220, 'TX_FRAUD_SCENARIO']=1
    nb_frauds_scenario_1=transactions_df.TX_FRAUD.sum()
    print("Number of frauds from scenario 1: "+str(nb_frauds_scenario_1))
    
    # Scenario 2
    for day in range(transactions_df.TX_TIME_DAYS.max()):
        
        compromised_terminals = terminal_profiles_table.TERMINAL_ID.sample(n=2, random_state=day)
        
        compromised_transactions=transactions_df[(transactions_df.TX_TIME_DAYS>=day) & 
                                                    (transactions_df.TX_TIME_DAYS<day+28) & 
                                                    (transactions_df.TERMINAL_ID.isin(compromised_terminals))]
                            
        transactions_df.loc[compromised_transactions.index,'TX_FRAUD']=1
        transactions_df.loc[compromised_transactions.index,'TX_FRAUD_SCENARIO']=2
    
    nb_frauds_scenario_2=transactions_df.TX_FRAUD.sum()-nb_frauds_scenario_1
    print("Number of frauds from scenario 2: "+str(nb_frauds_scenario_2))
    
    # Scenario 3
    for day in range(transactions_df.TX_TIME_DAYS.max()):
        
        compromised_customers = customer_profiles_table.CUSTOMER_ID.sample(n=3, random_state=day).values
        
        compromised_transactions=transactions_df[(transactions_df.TX_TIME_DAYS>=day) & 
                                                    (transactions_df.TX_TIME_DAYS<day+14) & 
                                                    (transactions_df.CUSTOMER_ID.isin(compromised_customers))]
        
        nb_compromised_transactions=len(compromised_transactions)
        
        
        random.seed(day)
        index_fauds = random.sample(list(compromised_transactions.index.values),k=int(nb_compromised_transactions/3))
        
        transactions_df.loc[index_fauds,'TX_AMOUNT']=transactions_df.loc[index_fauds,'TX_AMOUNT']*5
        transactions_df.loc[index_fauds,'TX_FRAUD']=1
        transactions_df.loc[index_fauds,'TX_FRAUD_SCENARIO']=3
        
                             
    nb_frauds_scenario_3=transactions_df.TX_FRAUD.sum()-nb_frauds_scenario_2-nb_frauds_scenario_1
    print("Number of frauds from scenario 3: "+str(nb_frauds_scenario_3))
    
    return transactions_df                 


def generate_dataset_default(n_customers = 10000, n_terminals = 1000000, nb_days=90, start_date="2018-04-01", r=5):
    
    start_time=time.time()
    customer_profiles_table = generate_customer_profiles_table(n_customers, random_state = 0)
    print("Time to generate customer profiles table: {0:.2}s".format(time.time()-start_time))
    
    start_time=time.time()
    terminal_profiles_table = generate_terminal_profiles_table(n_terminals, random_state = 1)
    print("Time to generate terminal profiles table: {0:.2}s".format(time.time()-start_time))
    
    start_time=time.time()
    x_y_terminals = terminal_profiles_table[['x_terminal_id','y_terminal_id']].values.astype(float)
    customer_profiles_table['available_terminals'] = customer_profiles_table.apply(lambda x : get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=r), axis=1)
    # With Pandarallel
    #customer_profiles_table['available_terminals'] = customer_profiles_table.parallel_apply(lambda x : get_list_closest_terminals(x, x_y_terminals=x_y_terminals, r=r), axis=1)
    customer_profiles_table['nb_terminals']=customer_profiles_table.available_terminals.apply(len)
    print("Time to associate terminals to customers: {0:.2}s".format(time.time()-start_time))
    
    start_time=time.time()
    transactions_df=customer_profiles_table.groupby('CUSTOMER_ID').apply(lambda x : generate_transactions_table(x.iloc[0], nb_days=nb_days)).reset_index(drop=True)
    print("Time to generate transactions: {0:.2}s".format(time.time()-start_time))
    
    # Sort transactions chronologically
    transactions_df=transactions_df.sort_values('TX_DATETIME')
    # Reset indices, starting from 0
    transactions_df.reset_index(inplace=True,drop=True)
    transactions_df.reset_index(inplace=True)
    # TRANSACTION_ID are the dataframe indices, starting from 0
    transactions_df.rename(columns = {'index':'TRANSACTION_ID'}, inplace = True)
    # transactions_df = add_frauds(customer_profiles_table, terminal_profiles_table, transactions_df)
    return (customer_profiles_table, terminal_profiles_table, transactions_df)
    





    
def execute(commands):
    data_base_connection = GraphDatabase.driver(uri = "bolt://localhost:7687", auth=("neo4j", "1234"))
    session = data_base_connection.session()    
    for c in commands:
        session.run(c)


    
    
def generate_CSV(n_customers = 100, n_terminals = 100, nb_days=50, start_date="2022-02-20", r=10):
    customer_profiles_table, terminal_profiles_table, transactions_df=generate_dataset_default(n_customers,n_terminals,nb_days,start_date,r)
    customer_profiles_table.to_csv("customers.csv",index=False)
    terminal_profiles_table.to_csv("terminals.csv",index=False)
    transactions_df.to_csv("transactions.csv",index=False)
    return (customer_profiles_table, terminal_profiles_table, transactions_df)
 
def hour_map(x):
    if (x>=0) & (x<6):
        period = 'night'
    elif (x>=6) & (x<12):
        period = 'morning'
    elif (x>=12) & (x<18):
        period = 'afternoon'
    else:
        period = 'evening'
    return period

def extract_period(datetimes):
    return datetimes.apply(lambda x:hour_map(x.hour))

def extend_transactions(df):

    start_time=time.time()
    
    df=df.assign(TX_PERIOD=lambda x: extract_period(x.TX_DATETIME))
    
    typep = ["high-tech", "food", "clothing", "consumable", "other"]
    
    li = [None]*df.size
    li = [random.randint(0,4) for x in li]
    li = [typep[x] for x in li]
    df=df.assign(TX_PRODUCT_TYPE=lambda x:pd.Series(li))
    print("Time to extend transactions with 'tipe of product' and 'period of day': {0:.2}s".format(time.time()-start_time))

    return df
    

    
def load_CSV():
    c1= """ CALL apoc.load.csv('/Users/pietrobarone/Documents/UniMI/DBMS/Progetto/customers.csv') yield map
            CALL apoc.create.node(["Customer"],{customer_id:map.CUSTOMER_ID, x_customer_id:  map.x_customer_id , y_customer_id: map.y_customer_id, mean_amount: map.mean_amount, std_amount: map.std_amount, mean_nb_tx_per_day: map.mean_nb_tx_per_day}) YIELD node 
            return count(*)"""
    c2= """ CALL apoc.load.csv('/Users/pietrobarone/Documents/UniMI/DBMS/Progetto/terminals.csv') yield map
            CALL apoc.create.node(["Terminal"],{terminal_id:map.TERMINAL_ID, x_terminal_id:  map.x_terminal_id , y_terminal_id: map.y_terminal_id}) YIELD node 
            return count(*)"""
    i1= """ CREATE  INDEX customer_index 
            FOR (c:Customer)
            ON (c.customer_id)"""
    i2= """ CREATE  INDEX terminal_index 
            FOR (t:Terminal)
            ON (t.terminal_id)"""
    c3= """ CALL apoc.load.csv('/Users/pietrobarone/Documents/UniMI/DBMS/Progetto/transactions.csv') yield map
            MATCH (c:Customer),(t:Terminal)
            WHERE c.customer_id=map.CUSTOMER_ID and t.terminal_id=map.TERMINAL_ID
            CALL apoc.create.relationship(c,"Transaction",{transaction_id:map.TRANSACTION_ID,tx_time_seconds:map.TX_TIME_SECONDS, tx_time_days:map.TX_TIME_DAYS,customer_id:map.CUSTOMER_ID,terminal_id:map.TERMINAL_ID,tx_amount:map.TX_AMOUNT, tx_datetime:datetime({epochmillis: apoc.date.parse(map.TX_DATETIME, "ms", "yyyy-MM-dd HH:mm:ss")}), tx_fraud:map.TX_FRAUD},t) YIELD rel
            return count(*)"""
    i3= """ CREATE  INDEX transaction_index 
            FOR (t:Transaction)
            ON (t.transaction_id)"""
            
    start_time=time.time()
    execute([c1,c2,i1,i2,c3,i3])
    print("Time to load CSV into DB: {0:.2}s".format(time.time()-start_time))


def clear_DB():
    c1="""drop index ON:Terminal(terminal_id)"""
    c2="""drop index ON:Customer(customer_id)"""
    c3="""DROP INDEX transaction_index"""
    c4="""match (n) detach delete n"""


    execute([c1,c2,c3,c4])

def updateDB(df):
    c=  """ CALL apoc.load.csv('/Users/pietrobarone/Documents/UniMI/DBMS/Progetto/extended_transactions.csv') yield map
            MATCH ()-[t:Transaction {transaction_id: map.TRANSACTION_ID}]-()
            set t.tx_period =map.TX_PERIOD, t.tx_product_type =map.TX_PRODUCT_TYPE"""
    
    start_time=time.time()
    execute([c])
    print("Time update DB: {0:.2}s".format(time.time()-start_time))

    
    
if __name__ == "__main__":
    clear_DB()
    customer_profiles_table, terminal_profiles_table, transactions_df=generate_CSV(1000,1000,5)
    
    load_CSV()
    
    # extend dataframe
    transactions_df = extend_transactions(transactions_df)
    
    transactions_df.to_csv("extended_transactions.csv",index=False)
    updateDB(transactions_df)
    
    s1=os.path.getsize("transactions.csv")
    s2=os.path.getsize("customers.csv")
    s3=os.path.getsize("terminals.csv")
    print(str((s1+s2+s3)*0.000001 )+" MB")
    
    
 

    
    

