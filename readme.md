#Introduction
This is a vi-ba translation project

#How to
1. Download checkpoints from <> to _checkpoints_ folder:   
.   
├ _config.yml   
├ checkpoints   
...├ dictionary   
......├ dict-synonymaugment.txt   
......└ dict-synonymaugment-accent.txt   
...├ dictionary_translate    
......├ data     
......└ dictionary    
.........├ bana_0504_w.txt   
.........└ vi_0504_w.txt       
...├ loan_former    
...├ phobert_fused    
...├ transformers   
...└ vi_ba_bart_pho  
1. Start VNCoreNLP:   
```vncorenlp -Xmx2g vncorenlp/VnCoreNLP-1.1.1.jar -p 9000 -a "wseg,pos,ner,parse"```

2. 