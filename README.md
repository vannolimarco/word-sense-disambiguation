# Welcome to Vannoli Marco github 
The repository is dedicated to the state-of-art of NLP, in this case the word sense disanbiguation. Inside it, you will find the code implemented and results about project. It has been implemented using babelnet and wordnet data.
# Report about it
The report producted by my self you can find inside the repository. Enjoy!
# Info about me
I am a student at the faculty of Artificial Inteligent and robotics of Sapienza (Rome). I am originally from Subiaco (RM) and I am passionate about the world of robotics and artificial intelligence, I am always active and enthusiastic when there is a new project to do, like this one!

## Support or Contact
my email: vannolimarco@hotmail.it, thanks!


## skeleton of project
      |_code       #all code 
        |_ pathconfig.py #file py to manage all paths
        |_ predict.py #code for predictions
        |_ preprocessing.py #code which contains class needs to preprocessing
        |_ run_gold_evaluation.py #file py which was used to create gold file for evaluation (only read, not run)
        |_ run_preprocessing.py #file py which was used to create preprocessing phase (only read, not run)
      |_resources # mappings are in here. you should place any additional resource (e.g. trained models) in here
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


## information
inside the folder resources, can find two sub folder:
evaluations and predictions, in evaluation there are all 
file gold used to evaluate the score F1 for each evalaution datasets.
in predictions there are all predictions obtained with the best model.
in teh same forder resources can find the file for model: model.json and
weights.h5; besiedes can find the file to load the model of words emebeddings ttrained. The same forlder 
includes the mapping babelnet-domains-lexnames provided.
inside the folder code there are all code; the code run_gold_evaluation.py
and run_preprocessing are only read because many file were deleted from repository.

 

```

