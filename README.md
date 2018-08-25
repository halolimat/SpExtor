# SpExtor: Sparse Entity Extractor

This is the implementation of the method in our COLING 2018 paper "A Practical Incremental Learning Framework For Sparse Entity Extraction".

## Slides
[COLING 2018 Slides](https://link.hussein.space/SpExtor-Slides)

## Data
Annotated Data mentioned in the paper in addition to many other datasets can be downloaded from [This Link](http://link.hussein.space/NERData).

## Source Tree

    .
    └── src
        ├── main  
        │   ├── java 
        │   │   ├── ActiveLearning.java       :  implements active learning
        │   │   ├── EntityLearning.java       :  core code
        │   │   ├── EntitySetExpansion.java   :  implementation of the entity set expansion method
        │   │   └── FeatureFactory.java       :  feature factory for the entity set expansion method
        │   │
        │   └── resources
        │       └── wordnet.dict    : contains WordNet dictionaries files
        │
        └── test 
            ├── java 
            │   ├── Dataset.java    : prepares the testing data
            │   └── main.java       : main testing class to run the full code
            │   
            └── resources   : contains the gold training and testing data
    
## Citing

If you do make use of SpExtor or any of its components please cite the following publication:

    Hussein S. Al-Olimat, Steven Gustafson, Jason Mackay, Krishnaprasad Thirunarayan, and Amit Sheth. 2018. 
    A practical incremental learning framework for sparse entity extraction. In Proceedings of the 27th Internationl
    Conference on Computational Linguistics (COLING 2018), pages 700–710. Association for Computational Linguistics.
    
    @InProceedings{C18-1059,
      author = 	"Al-Olimat, Hussein S.
                and Gustafson, Steven
                and Mackay, Jason
                and Thirunarayan, Krishnaprasad
                and Sheth, Amit",
      title = "A Practical Incremental Learning Framework For Sparse Entity Extraction",
      booktitle = "Proceedings of the 27th International Conference on Computational Linguistics",
      year = "2018",
      publisher = "Association for Computational Linguistics",
      pages = "700--710",
      location = "Santa Fe, New Mexico, USA",
      url = "http://aclweb.org/anthology/C18-1059"
    }


We would also be very happy if you provide a link to the github repository:

    ... Sparse Entity Extractor tool (SpExtor)\footnote{
        \url{https://github.com/halolimat/SpExtor}
    }