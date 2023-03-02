# NEW GENERATION DATA MODELS AND DBMSS' Project
fraud detection nosql db

The project has been developed using a graph database. In particular neo4j.

UML class diagram
Thinking of a logical model that adapts to the project request, it is easy to define a first draft of the model through two main subjects (customer and terminal) connected to each other through a relationship (transaction). The peculiarity of the transactions is that they contain peculiar data that exists only when the aforementioned transaction is created. For this reason it makes sense to define the transaction using the "reified relationship" type provided by UML.

Given the simplicity of the model, it is not necessary to define additional constraints in addition to those that can be specified via UML.

Immagine UML

Logical model
The logical model is realized, as already mentioned, in neo4j. It directly depends on the UML model and from this you can see the similarities. There are two labels:

Customer (with related properties)
Terminal (with related properties)
Mirroring the UML model there is only one relationship:

Transaction (with related properties)
Why graph database
The main reason that convinced me to use this model comes from query c. In particular, the request for co-customer-relationships CC of degree k is a chained relationship that is intuitively mappable to a path of a graph. Neo4j being the state of the art for graph db seemed the best solution. In fact, the query in question is the most complex in computational terms, and by exploiting the structure of neo4j, both the structure of the query itself and its execution are more natural.

Immagine modello logico

The script
c. The description of the script for loading the datasets in the chosen NOSQL system

d. The description of the scripts for the implementation of the required operations

e. A discussion of the performances obtained by the execution of the operations on the three considered datasets. Please, discuss the eventual application of patterns for improving the performances of the considered operations.
