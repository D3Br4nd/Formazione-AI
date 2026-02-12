# Modello di classificazione dei pezzi difettosi in produzione

## Descrizione del progetto

L'azienda **AutomaParts S.p.A.** è un fornitore tier-1 per il settore automotive che produce componenti meccanici di precisione per sistemi sterzo e sospensioni. Il progetto ha l'obiettivo di sviluppare, usando Python e consegnando un notebook Google Colab, un modello di machine learning supervisionato per distinguere pezzi conformi da pezzi difettosi in linea di produzione. Il modello dovrà supportare decisioni operative per ridurre scarti, aumentare la qualità e migliorare i tempi di ispezione.

Il lavoro è pensato per essere applicabile a dati raccolti da celle di produzione reali (misure dimensionali, parametri di processo, segnali di sensori, informazioni di linea e operatore) e dovrà includere analisi del dataset, gestione dei dati mancanti, preparazione delle variabili, addestramento e valutazione di modelli (ad es. Decision Tree, Random Forest, Logistic Regression) con valutazione della capacità di generalizzazione.

Caso d'uso: in una linea di assemblaggio di bracci trasversali vengono misurate più caratteristiche per ogni pezzo (es. diametri, planaritá, coppia, temperatura processo). Alcuni pezzi non rispettano tolleranze e causano fermi macchina o richiami cliente. Il modello dovrà predire la probabilità che un pezzo sia difettoso al termine del processo di controllo in linea, permettendo:

*   scarto automatico o blocco in ispezione finale;
*   instradamento verso un controllo 100% quando la probabilità è incerta;
*   analisi delle cause principali dei difetti per interventi di processo.

Importanza per l'azienda: riduzione dei costi di scarto e rilavorazione, aumento del First Pass Yield, diminuzione del rischio di non conformità verso OEM, e migliore efficienza della linea.

Valore aggiunto: tracciabilità dei motivi di scarto aggregata per macchina/lotto/operatore, supporto alle decisioni manutentive e potenziale integrazione con sistemi MES per azioni automatizzate.

## Dataset

File: parts_production_data.csv (scaricabile dal seguente link: [https://github.com/Profession-AI/progetti-ml/blob/main/Modello%20di%20classificazione%20di%20pezzi%20difettosi%20in%20produzione/parts_production_data.csv](https://github.com/Profession-AI/progetti-ml/blob/main/Modello%20di%20classificazione%20di%20pezzi%20difettosi%20in%20produzione/parts_production_data.csv))

Colonne: 1. part_id: identificativo univoco del pezzo 2. production_timestamp: data/ora di produzione (può essere usata per aggregazioni temporali) 3. line_id: identificativo della linea/cella produttiva 4. station_id: identificativo della stazione di misura 5. operator_id: identificativo operatore (anonimizzato se necessario) 6. measure_diam_mm: misura dimensionale (mm) 7. measure_length_mm: misura longitudinale (mm) 8. flatness_mm: planaritá superficiale (mm) 9. torque_Nm: valore di coppia registrato 10. surface_roughness_Ra: rugosità superficiale 11. temp_process_C: temperatura del processo (°C) 12. vibration_level: livello di vibrazione misurato 13. cycle_time_s: tempo ciclo della macchina 14. material_batch: lotto materia prima 15. visual_inspection_score: punteggio ispezione visiva (se disponibile) 16. defect_label: etichetta binaria (0 = conforme, 1 = difettoso)

Nota: il dataset potrà contenere variabili numeriche, categoriche e possibilmente feature derivate da sensori o ispezioni automatiche.

## Obiettivi del progetto

Il progetto richiede la realizzazione delle seguenti attività a livello di requisito (alto livello):

1.  Analisi esplorativa del dataset
    
    *   Esplorazione delle variabili, distribuzioni e correlazioni a livello descrittivo.
    *   Individuazione di eventuali squilibri nella classe target (difettosi vs conformi).
    *   Visualizzazioni riassuntive utili per stakeholder (es. tassi di difettosità per linea/lotto).
2.  Pulizia dei dati e gestione dei valori mancanti
    
    *   Identificazione delle variabili con valori mancanti e valutazione dell'impatto sul dataset.
    *   Definizione di criteri per trattare i record incompleti o le feature con alta percentuale di missing.
    *   Trattamento coerente dei dati anomali e outlier a fini analitici.
3.  Preparazione delle variabili
    
    *   Codifica delle variabili categoriche in forma utilizzabile dai modelli.
    *   Normalizzazione/scaling delle variabili numeriche per consentire una corretta comparazione.
    *   Creazione di nuove feature derivate se ritenuto rilevante (ad es. indici aggregati per lotto o tempo).
4.  Suddivisione dei dati e valutazione della capacità di generalizzazione
    
    *   Definizione di dataset di training e test (o equivalenti) per misurare la generalizzazione.
    *   Applicazione di metodologie di valutazione per stimare il comportamento su dati non visti.
    *   Verifica della stabilità delle prestazioni in scenari rilevanti per produzione.
5.  Addestramento e confronto di modelli supervisati
    
    *   Addestramento di almeno tre tipologie di modelli di riferimento: Decision Tree, Random Forest e Logistic Regression.
    *   Confronto delle performance e analisi qualitativa dei risultati (vantaggi/svantaggi di ciascun approccio per contesto industriale).
6.  Valutazione delle performance
    
    *   Misurazione delle metriche di valutazione principali (es. accuracy, precision, recall, F1 score, ROC-AUC).
    *   Analisi delle implicazioni dei falsi negativi (pezzi difettosi classificati come conformi) e dei falsi positivi dal punto di vista operativo e di costo.
    *   Presentazione di una tabella riassuntiva con metriche per ciascun modello testato.
7.  Interpretabilità e utilità operativa
    
    *   Analisi ad alto livello delle feature più informative per il modello (per supportare azioni correttive).
    *   Valutazione dell'applicabilità del modello in produzione e delle condizioni in cui la predizione è affidabile.
8.  Conclusioni e raccomandazioni
    
    *   Sintesi dei risultati con indicazione del modello preferibile e dei limiti riscontrati.
    *   Raccomandazioni operative per l'implementazione in linea (es. soglie di intervento, monitoraggio continuità delle prestazioni).
    *   Proposte di ulteriori analisi o dati da raccogliere per migliorare le prestazioni.

## Requisiti di qualità e vincoli

*   Documentazione chiara e sintetica delle scelte effettuate: ipotesi sul dato, metriche scelte, criteri di valutazione.
*   Attenzione alla privacy: eventuali identificativi sensibili (operator\_id, etc.) devono essere anonimizzati o trattati secondo policy aziendali.
*   Considerare la robustezza del modello rispetto a variazioni di processo e possibili bias introdotti dai dati storici.

## Criteri di valutazione

*   Completezza dell’analisi esplorativa e chiarezza nella descrizione del dataset.
*   Corretta gestione dei dati mancanti e delle variabili non informative.
*   Coerenza del processo di preparazione delle variabili e della separazione train/test.
*   Qualità del confronto tra i modelli: metriche rilevanti, interpretazione e considerazioni sui falsi negativi/falsi positivi.
*   Chiarezza delle conclusioni e delle raccomandazioni operative per AutomaParts S.p.A.

Il progetto mira a fornire a AutomaParts S.p.A. uno strumento decisionale per migliorare la qualità di produzione e ridurre i costi legati a pezzi non conformi. Oltre alla mera accuratezza predittiva, la rilevanza pratica del progetto risiede nella valutazione dell'impatto operativo dei falsi negativi, nella tracciabilità delle cause e nella proposta di raccomandazioni per l'integrazione con i processi esistenti.

* * *

**Modalità di consegna:**

Link pubblico a notebook di Google Colab

## Politica di massimo impegno

In ProfessionAI crediamo fermamente che il valore delle competenze sia superiore a quello dei titoli accademici. In un futuro in cui l'intelligenza artificiale supporterà sempre più le attività professionali, a distinguersi saranno coloro che possiedono competenze solide, approfondite e di alto livello.

Proprio perché abbiamo profondamente a cuore il successo professionale dei nostri studenti, chiediamo a tutti il massimo impegno e la massima onestà intellettuale nello svolgimento dei progetti:

*   Ogni progetto può essere consegnato fino a **3 volte gratuitamente**. Eventuali consegne aggiuntive comportano un costo di **50€ IVA inclusa** ciascuna.
*   Il progetto deve essere **ben strutturato e documentato**. Per elaborati di coding, analisi dati o machine learning non è sufficiente consegnare il solo codice: è necessario includere una spiegazione chiara del problema affrontato, delle soluzioni adottate e delle conclusioni raggiunte.
*   È **severamente vietato** l'utilizzo di strumenti come ChatGPT, Claude, Gemini o soluzioni simili di AI generativa. È invece consentito - e consigliato - l'uso di **ProfAI**, sviluppato per supportare lo studio senza sostituire lo studente.
*   Non è ammessa la consegna di progetti realizzati, totalmente o in parte, da altre persone.
*   È consentito l'utilizzo di risorse esterne (es. blog, Stack Overflow, GitHub, Kaggle e servizi simili) per reperire informazioni utili. In questi casi, è obbligatorio includere una sezione finale **“Riferimenti”** con i link alle fonti utilizzate.

Qualora il docente nutra dubbi sull'effettiva realizzazione autonoma del progetto, potrà richiedere un **colloquio orale di approfondimento in videocall**. L'eventuale mancato superamento del colloquio comporterà l'invalidamento delle correzioni gratuite residue per tale progetto; per una nuova consegna sarà quindi necessario versare il contributo di **50€ IVA inclusa**.

Ultimo aggiornamento dei termini

  
 

eser