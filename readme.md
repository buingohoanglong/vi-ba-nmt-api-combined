# Introduction   
This is a vi-ba translation project

# How to
1. Download checkpoints to _checkpoints_ folder:   
link1: https://drive.google.com/drive/folders/1-0RNqTFa596aH4BJCi7Pz4_hdkDGnLiu?usp=sharing   
link2: https://drive.google.com/drive/folders/1ZKbl1-DmJ-waIrLwqIlTHYL4SWM772Qo  

- Copy link1 to vi_ba_bartpho
- Copy link2 to loan_former, phobert_fused, transformers

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

2. Prepare VnCoreNLP: <br/>
```cd model/transformerbertpgn``` <br/>
```./prepare_vncorenlp.sh```

3. Start VNCoreNLP:   
```vncorenlp -Xmx2g vncorenlp/VnCoreNLP-1.1.1.jar -p 9000 -a "wseg,pos,ner,parse"```

4. Start API Server:   
```PYTHONPATH=./ python api/translation_api.py```

# References
Pham Quoc Nguyen. https://github.com/PhamNguyen97/ViBaCombineModel
