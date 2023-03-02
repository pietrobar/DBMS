# Progetto NEW GENERATION DATA MODELS AND DBMSS

>Questo progetto riguarda la ricerca e l'implementazione soluzione migliore per gestire dati bancari in un sistema di rilevamento di frodi con carta di credito.

Riferimento al [testo del progetto](https://github.com/pietrobar/DBMS/blob/master/Project-fraudDetection-vers1.pdf)

Il progetto è stato sviluppato utilizzando un database a grafo. In particolare tramite neo4j.

Diagramma delle classi UML
Pensando ad un modello logico che si adatti alla richiesta progettuale, è facile definire una prima bozza del modello attraverso due soggetti principali (cliente e terminale) collegati tra loro attraverso una relazione (transazione). La particolarità delle transazioni è che contengono dati peculiari che esistono solo quando viene creata la suddetta transazione. Per questo motivo ha senso definire la transazione utilizzando il tipo di "relazione reificata" fornito da UML.

Data la semplicità del modello, non è necessario definire ulteriori vincoli oltre a quelli specificabili tramite UML.

Ecco il diagramma definito:

![Screenshot 2023-03-02 alle 11 40 54](https://user-images.githubusercontent.com/34039242/222405636-8dd2d7cd-f74e-48fa-804e-dd16fb45b92f.jpg)


Modello logico
Il modello logico è realizzato, come già accennato, in neo4j. Dipende direttamente dal modello UML. Ci sono due etichette:

Cliente (con relative proprietà)
Terminale (con proprietà correlate)

Rispecchiando il modello UML c'è solo una relazione:
Transazione (con proprietà correlate)

Il modello logico ha una struttura del tutto simile al diagramma delle classi


Perché il database a grafo?
Il motivo principale che mi ha convinto ad utilizzare questo modello viene dalla query c. In particolare, la richiesta di co-clienti CC di grado k è una relazione concatenata intuitivamente mappabile ad un percorso di un grafo. Neo4j essendo lo stato dell'arte per i db a grafo sembrava la soluzione migliore. Infatti la query in questione è la più complessa in termini computazionali, e sfruttando la struttura di neo4j, sia la struttura della query stessa che la sua esecuzione risultano più efficienti.


Cosa contiene lo [script](https://github.com/pietrobar/DBMS/blob/master/loading_and_operations.py):

c. La descrizione dello script per caricare i dataset nel sistema NOSQL scelto

d. La descrizione degli script per l'implementazione delle operazioni richieste

e. Una discussione delle prestazioni ottenute dall'esecuzione delle operazioni sui tre dataset considerati. Si prega di discutere l'eventuale applicazione di schemi per migliorare le prestazioni delle operazioni considerate.
