# Necessary imports for this notebook
from email.policy import default
import os
from unicodedata import name
from neo4j import GraphDatabase

import numpy as np
import pandas as pd

import datetime
import time

import random

import threading, queue

from pandarallel import pandarallel
pandarallel.initialize()



import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid', {'axes.facecolor': '0.9'})

path="default/"
base="/Users/pietrobarone/Documents/UniMI/DBMS/Progetto/"#set global path

def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f'{func.__name__:<35} {"executed in":^20} {(time.time()-start_time):>.5f}s')
        return result
    return wrap_func


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



def generate_transactions_table(customer_profile, start_date = "2022-03-03", nb_days = 10):
    
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
    

@timer_func
def add_frauds(transactions_df):
    
    #random frauds
    transactions_df['TX_FRAUD']=random.randint(0,1) == 1
    return transactions_df  
               
def enthread(target, args):
    q = queue.Queue()
    def wrapper():
        q.put(target(*args))
    t = threading.Thread(target=wrapper)
    t.start()
    return q,t

def generate_dataset_default(n_customers = 10000, n_terminals = 1000000, nb_days=90, start_date="2018-04-01", r=5):
    
    
    q1,thread1 = enthread(target=generate_customer_profiles_table,args=(n_customers,0))
    start_time=time.time()
    #customer_profiles_table = generate_customer_profiles_table(n_customers, random_state = 0)
    
    terminal_profiles_table=None
    q2,thread2 = enthread(target=generate_terminal_profiles_table,args=(n_terminals,1))
    start_time=time.time()

    #terminal_profiles_table = generate_terminal_profiles_table(n_terminals, x = 1)
    
    thread1.join()
    customer_profiles_table=q1.get()

    print("Time to generate customer profiles table: {0:.10}s   Number of elements:".format(time.time()-start_time), len(customer_profiles_table))
    thread2.join()
    terminal_profiles_table=q2.get()
    print("Time to generate terminal profiles table: {0:.10}s   Number of elements:".format(time.time()-start_time), len(terminal_profiles_table))
    
    start_time=time.time()
    x_y_terminals = terminal_profiles_table[['x_terminal_id','y_terminal_id']].values.astype(float)
    #customer_profiles_table['available_terminals'] = customer_profiles_table.apply(lambda x : get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=r), axis=1)
    # With Pandarallel
    customer_profiles_table['available_terminals'] = customer_profiles_table.parallel_apply(lambda x : get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=r), axis=1)
    customer_profiles_table['nb_terminals']=customer_profiles_table.available_terminals.apply(len)
    print("Time to associate terminals to customers: {0:.10}s".format(time.time()-start_time))
    
    start_time=time.time()
    #transactions_df=customer_profiles_table.groupby('CUSTOMER_ID').apply(lambda x : generate_transactions_table(x.iloc[0], nb_days=nb_days)).reset_index(drop=True)
    # With Pandarallel
    transactions_df=customer_profiles_table.groupby('CUSTOMER_ID').parallel_apply(lambda x : generate_transactions_table(x.iloc[0], nb_days=nb_days)).reset_index(drop=True)
    print("Time to generate transactions:            {0:.10}s       Number of elements:".format(time.time()-start_time), len(transactions_df))
    
    # Sort transactions chronologically
    transactions_df=transactions_df.sort_values('TX_DATETIME')
    # Reset indices, starting from 0
    transactions_df.reset_index(inplace=True,drop=True)
    transactions_df.reset_index(inplace=True)
    # TRANSACTION_ID are the dataframe indices, starting from 0
    transactions_df.rename(columns = {'index':'TRANSACTION_ID'}, inplace = True)
    transactions_df = add_frauds(transactions_df)
    return (customer_profiles_table, terminal_profiles_table, transactions_df)
    





    
def execute(commands,data_base_connection):
    
    session = data_base_connection.session()    
    for c in commands:
        session.run(c)


    
    
def generate_CSV(n_customers = 100, n_terminals = 100, nb_days=50, start_date="2022-02-20", r=10):
    customer_profiles_table, terminal_profiles_table, transactions_df=generate_dataset_default(n_customers,n_terminals,nb_days,start_date,r)
    outdir = './'+path
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    customer_profiles_table.to_csv(path+"customers.csv",index=False)
    terminal_profiles_table.to_csv(path+"terminals.csv",index=False)
    transactions_df.to_csv(path+"transactions.csv",index=False)
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

@timer_func
def extend_transactions(df):    
    df=df.assign(TX_PERIOD=lambda x: extract_period(x.TX_DATETIME))
    
    typep = ["high-tech", "food", "clothing", "consumable", "other"]
    
    li = [None]*df.size
    li = [random.randint(0,4) for x in li]
    li = [typep[x] for x in li]
    df=df.assign(TX_PRODUCT_TYPE=lambda x:pd.Series(li))
    df.to_csv(path+"extended_transactions.csv",index=False)
    
@timer_func
def extend_transactions_base(connection):

    period = ['morning', 'afternoon', 'evening', 'night']
    typep = ['high-tech', 'food', 'clothing', 'consumable', 'other']
    
    session = connection.session()   
    transactions = session.run("MATCH ()-[t:Transaction]->() RETURN t.transaction_id").values()
    
    for i in range(0,len(transactions)):
        tx_period = period[random.randint(0,len(period)-1)]
        tx_product_type = typep[random.randint(0,len(typep)-1)]
        c="""   MATCH ()-[t:Transaction {transaction_id : """+str(transactions[i][0])+"""}]-() 
                SET t.tx_period = '"""+tx_period+"""', 
                t.tx_product_type = '"""+tx_product_type+"""'
            """
        session.run(c)
    

@timer_func
def load_CSV(connection):
    c1=""" CALL apoc.periodic.iterate('
            CALL apoc.load.csv(\\'"""+base+path+"""customers.csv\\') yield map as row return row','
            CALL apoc.create.node(["Customer"],
            {customer_id: toInteger(row.CUSTOMER_ID), 
            x_customer_id:  toFloat(row.x_customer_id) , 
            y_customer_id: toFloat(row.y_customer_id), 
            mean_amount: toFloat(row.mean_amount), 
            std_amount: toFloat(row.std_amount), 
            mean_nb_tx_per_day: toFloat(row.mean_nb_tx_per_day)}) YIELD node 
            return count(*)
        ', { parallel:true, concurrency:1000,batchSize:1000});"""
    c2= """ CALL apoc.periodic.iterate('
            CALL apoc.load.csv(\\'"""+base+path+"""terminals.csv\\') yield map as row return row','
            CALL apoc.create.node(["Terminal"],
            {terminal_id:   toInteger(row.TERMINAL_ID), 
            x_terminal_id:  toFloat(row.x_terminal_id) , 
            y_terminal_id:  toFloat(row.y_terminal_id)}) YIELD node 
            return count(*)
            ', { parallel:true, concurrency:1000,batchSize:1000});"""
    i1= """ CREATE CONSTRAINT customer_id FOR (c:Customer) REQUIRE c.customer_id IS UNIQUE"""
    i2= """ CREATE CONSTRAINT terminal_id FOR (t:Terminal) REQUIRE t.terminal_id IS UNIQUE"""
    c3= """ CALL apoc.periodic.iterate('
            CALL apoc.load.csv(\\'"""+base+path+"""transactions.csv\\') yield map as row return row','
            MATCH (c:Customer),(t:Terminal)
            WHERE c.customer_id=toInteger(row.CUSTOMER_ID) and t.terminal_id=toInteger(row.TERMINAL_ID)
            CALL apoc.create.relationship(c,"Transaction",
            {transaction_id:toInteger(row.TRANSACTION_ID),
            tx_time_seconds:toInteger(row.TX_TIME_SECONDS),
            tx_time_days:toInteger(row.TX_TIME_DAYS),
            customer_id:toInteger(row.CUSTOMER_ID),
            terminal_id:toInteger(row.TERMINAL_ID),
            tx_amount:toFloat(row.TX_AMOUNT),
            tx_datetime:datetime({epochmillis: apoc.date.parse(row.TX_DATETIME, "ms", "yyyy-MM-dd HH:mm:ss")}), 
            tx_fraud:toInteger(row.TX_FRAUD)},t) YIELD rel
            return count(*)
            ', { parallel:true, concurrency:10000,batchSize:1000});"""
    i3= """ CREATE CONSTRAINT transaction_id FOR (t:Transaction) REQUIRE t.transaction_id IS UNIQUE"""
            
    execute([c1,c2,i1,i2,c3,i3],connection)

@timer_func
def clear_DB(connection):
    c1="""DROP CONSTRAINT terminal_id IF EXISTS"""
    c2="""DROP CONSTRAINT customer_id IF EXISTS"""
    c3="""DROP CONSTRAINT transaction_id IF EXISTS"""
    c4="""  CALL apoc.periodic.iterate(
            'MATCH ()-[r]->() RETURN id(r) AS id', 
            'MATCH ()-[r]->() WHERE id(r)=id DELETE r', 
            {batchSize: 10000});"""
    c5="""  CALL apoc.periodic.iterate(
            'MATCH (n) RETURN id(n) AS id', 
            'MATCH (n) WHERE id(n)=id DELETE n', 
            {batchSize: 10000});"""

    execute([c1,c2,c3,c4,c5],connection)

@timer_func
def updateDB(connection):
    c=  """ CALL apoc.periodic.iterate('
            CALL apoc.load.csv(\\'"""+base+path+"""extended_transactions.csv\\') yield map as row return row','
            MATCH ()-[t:Transaction {transaction_id: toInteger(row.TRANSACTION_ID)}]-()
            set t.tx_period =row.TX_PERIOD, t.tx_product_type =row.TX_PRODUCT_TYPE
            ', { parallel:true, concurrency:10000,batchSize:1000});"""
    
    execute([c],connection)
  
@timer_func  
def fastUpdateDB(connection):
    d="""match ()-[r:Transaction]-() detach delete r"""
    c="""   CALL apoc.periodic.iterate('
            CALL apoc.load.csv(\\'"""+base+path+"""extended_transactions.csv\\') yield map as row return row','
            MATCH (c:Customer),(t:Terminal)
            WHERE c.customer_id=toInteger(row.CUSTOMER_ID) and t.terminal_id=toInteger(row.TERMINAL_ID)
            CALL apoc.create.relationship(c,"Transaction",
            {transaction_id:toInteger(row.TRANSACTION_ID),
            tx_time_seconds:toInteger(row.TX_TIME_SECONDS),
            tx_time_days:toInteger(row.TX_TIME_DAYS),
            customer_id:toInteger(row.CUSTOMER_ID),
            terminal_id:toInteger(row.TERMINAL_ID),
            tx_amount:toFloat(row.TX_AMOUNT),
            tx_datetime:datetime({epochmillis: apoc.date.parse(row.TX_DATETIME, "ms", "yyyy-MM-dd HH:mm:ss")}), 
            tx_fraud:toInteger(row.TX_FRAUD),
            tx_period:row.TX_PERIOD, 
            tx_product_type:row.TX_PRODUCT_TYPE},t) YIELD rel
            return count(*)
            ', { parallel:true, concurrency:10000,batchSize:1000});"""
    
    execute([d,c],connection)
    
    
@timer_func
def addFraud_asRequested(connection):
    c= """  call apoc.periodic.iterate(
            "match ()-[t:Transaction]->(:Terminal)
                    with datetime() as today,t
                    where toInteger(apoc.date.format(today.epochMillis-t.tx_datetime.epochMillis,\\"ms\\", \\"dd\\"))<31
                    with t.terminal_id as tid,avg(t.tx_amount) as avg return tid,avg",
            "match ()-[t:Transaction]-()
            where t.terminal_id = tid and t.tx_amount>avg+avg/2
            SET t.tx_fraud=1
            return t as fraudolentTransaction",
            { parallel:true, concurrency:1000,batchSize:100}
            )"""
    execute([c],connection)

    
@timer_func
def set_buying_friends_iterate(connection):
    
    c="""CALL apoc.periodic.iterate(
    "MATCH (a)-[t:Transaction]-(b) Return distinct(t.tx_product_type) as type",
    "MATCH (c1)-[t1:Transaction{tx_product_type: type}]->(t)<- 
    [t2:Transaction{tx_product_type: type}]-(c2)
    WITH c1, c2,t, COUNT(DISTINCT t1) AS n1, COUNT(DISTINCT t2) AS n2 
    WHERE c1.customer_id > c2.customer_id AND n1 > 3 AND n2 > 3
    MERGE (c1)-[:BUYING_FRIEND]->(c2)",
        { parallel:true, concurrency:1000,batchSize:100})
    """
    execute([c],connection)
  
  
        
@timer_func
def set_buying_friends(connection):
    session = connection.session()
    for t in ["high-tech", "food", "clothing", "consumable", "other"]:
        c="""   
                MATCH (c1)-[t1:Transaction{tx_product_type: '"""+t+"""'}]->(t)<- 
                [t2:Transaction{tx_product_type: '"""+t+"""'}]-(c2)
                WITH c1, c2,t, COUNT(DISTINCT t1) AS n1, COUNT(DISTINCT t2) AS n2 
                WHERE c1.customer_id > c2.customer_id AND n1 > 3 AND n2 > 3
                MERGE (c1)-[:BUYING_FRIEND]->(c2)
        """
        session.run(c)



    
@timer_func
def query_a(connection):
    c="""   with datetime() as today
            match (c:Customer)-[t:Transaction]->(:Terminal)
            where t.tx_datetime.year = today.year and t.tx_datetime.month = today.month
            return c.customer_id, t.tx_time_days,sum(t.tx_amount) order by c.customer_id, t.tx_time_days"""
    execute([c],connection)
  
@timer_func  
def query_b(connection):
    c="""   match ()-[t:Transaction]->(:Terminal)
            with datetime() as today,t
            where toInteger(apoc.date.format(today.epochMillis-t.tx_datetime.epochMillis,"ms", "dd"))<31
            with t.terminal_id as tid,avg(t.tx_amount) as avg
            CALL{
                with tid,avg
                match ()-[t:Transaction]-()
                where t.terminal_id = tid and t.tx_amount>(avg+avg/2)
                return t as fraudolentTransaction
            }
            return fraudolentTransaction"""
    execute([c],connection)
   
@timer_func 
def query_c(connection, customer_id=4, degree=3):
    degree=degree*2
    c="""   MATCH path=(a:Customer{customer_id:"""+str(customer_id)+"""})-[*"""+str(degree)+"""]-(b:Customer) 
            where  apoc.coll.duplicates(NODES(path)) = [] 
            return DISTINCT b limit 50"""
    execute([c],connection)
    
@timer_func
def query_e(connection):
    c1="""  MATCH ()-[t:Transaction]->()
            RETURN t.tx_period, count(t) as number_of_transactions, 
            sum(t.tx_fraud) as number_of_fraud_transactions"""
    execute([c1],connection)  



def print_sizes():
    s1=os.path.getsize(path+"transactions.csv")
    s2=os.path.getsize(path+"customers.csv")
    s3=os.path.getsize(path+"terminals.csv")
    print("-"*40)
    print(f'{"transactions.csv ":<20} {str((s1)*0.000001 ):^6} MB ')
    print(f'{"customers.csv ":<20} {str((s2)*0.000001 ):^6} MB')
    print(f'{"terminals.csv ":<20} {str((s3)*0.000001 ):^6} MB')
    print(f'{"TOTAL: ":<20} {str((s1+s2+s3)*0.000001 ):^6} MB')
    print("-"*40)
    
@timer_func
def create_model(p, data_base_connection, n_customers = 100, n_terminals = 100, nb_days=50):
    global path
    path=p
    
    
    #Generate data and convert it into CSV
    customer_profiles_table, terminal_profiles_table, transactions_df=generate_CSV(n_customers,n_terminals,nb_days)
    
    print_sizes()
    
    #LOAD CSV into DB
    load_CSV(data_base_connection)
    
    query_a(data_base_connection)
    query_b(data_base_connection)
    query_c(data_base_connection)
    
    # extend dataframe
    extend_transactions(transactions_df)#crea il csv
    updateDB(data_base_connection)
    # fastUpdateDB(data_base_connection)
    
    #extend_transactions_base(data_base_connection)
    
    set_buying_friends(data_base_connection)
    
    query_e(data_base_connection)

    
    
    
    


if __name__ == "__main__":
    data_base_connection=GraphDatabase.driver(uri = "bolt://localhost:7687", auth=("neo4j", "1234"))
    clear_DB(data_base_connection)
    create_model("small/",data_base_connection,9000,1000,35)
    clear_DB(data_base_connection)
    create_model("medium/",data_base_connection,5000,10000,160)
    clear_DB(data_base_connection)
    create_model("big/",data_base_connection,10000,10000,160)
    data_base_connection.close()
    

    
    
    
 

    
    

