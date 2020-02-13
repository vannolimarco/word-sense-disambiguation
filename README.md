# NLP Homework 3

## skeleton
- code       #all code 
  |_ pathconfig.py #file py to manage all paths
  |_ predict.py #code for predictions
  |_ preprocessing.py #code which contains class needs to preprocessing
  |_ run_gold_evaluation.py #file py which was used to create gold file for evaluation (only read, not run)
  |_ run_preprocessing.py #file py which was used to create preprocessing phase (only read, not run)
- resources # mappings are in here. you should place any additional resource (e.g. trained models) in here
  |__ babelnet2lexnames.tsv  # bnids to lexnames
  |__ babelnet2wndomains.tsv # bnids to WordNet domains labels
  |__ babelnet2wordnet       # bnids to WordNet offsets
  |__ evaluations
     |__ semeval2007
        |__ babelnet.gold.key.txt
        |__ domains.gold.key.txt
        |__ lexnames.gold.key.txt
     |__ semval2013
        |__ babelnet.gold.key.txt
        |__ domains.gold.key.txt
        |__ lexnames.gold.key.txt
     |__ semval2015
        |__ babelnet.gold.key.txt
        |__ domains.gold.key.txt
        |__ lexnames.gold.key.txt
     |__ senseval2
        |__ babelnet.gold.key.txt
        |__ domains.gold.key.txt
        |__ lexnames.gold.key.txt
     |__ senseval3
        |__ babelnet.gold.key.txt
        |__ domains.gold.key.txt
        |__ lexnames.gold.key.txt
  |__ predictions
     |__ semeval2007
        |__ babelnet.key
        |__ domains.key
        |__ lexnames.key
     |__ semval2013
        |__ babelnet.key
        |__ domains.key
        |__ lexnames.key
     |__ semval2015
        |__ babelnet.key
        |__ domains.key
        |__ lexnames.key
     |__ senseval2
        |__ babelnet.key
        |__ domains.key
        |__ lexnames.key
     |__ senseval3
       |__ babelnet.key
        |__ domains.key
        |__ lexnames.key
  |__ model.json
  |__ weights.h5
  |__ model_word_embeddings
  |__ README-md this file
-README.md #this file
-report.pdf #the report 

  
 #information
 inside the folder resources, can find twosub folder:
 evaluations and predictions, in evaluation there are all 
 file gold used to evaluate the score F1 for each evalaution datasets.
 in predictions there are all predictions obtained with the best model.
 in teh same forder resources can find the file for model: model.json and
 weights.h5; besiedes can find the file to load the model of words emebeddings ttrained. The same forlder 
 includes the mapping babelnet-domains-lexnames provided.
 inside the folder code there are all code; the code run_gold_evaluation.py
 and run_preprocessing are only read because many file were deleted from repository.

 

```

