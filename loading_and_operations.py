from neo4j import GraphDatabase

import pandas as pd


from timer_module import timer_func
from generate_data import *

import random


path="default/"
base="/Users/pietrobarone/Documents/UniMI/DBMS/Progetto/"#set global path



def execute(commands,data_base_connection):
    
    session = data_base_connection.session()    
    for c in commands:
        session.run(c)

 
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

    period = ["morning", "afternoon", "evening", "night"]
    typep = ["high-tech", "food", "clothing", "consumable", "other"]
    
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
    

    
    
    
 

    
    

