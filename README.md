## DAY 1
   用langchain＋Olloma实现，对数据进行了加载，进行了文档切分为chunk处理，主要用到了：父子切割和语义切割，这里语义切分用到了embedding 模型（omic-embed-text）对句子之间进行相似度的计算，从而提高了切分的准确性。
   问题1.对于PDF文件中出现的表格处理不当，这里用到的pypdf会让表格乱序。
   <img width="819" height="355" alt="image" src="https://github.com/user-attachments/assets/1697bfb7-af8f-4012-a5ad-ce7444f52b82" />
   <img width="826" height="384" alt="image" src="https://github.com/user-attachments/assets/8baacf00-2af6-43d7-9d81-6cf640ed18c0" />
   问题2.这里embedding模型的选择会影响我的结果吗？
   <img width="846" height="451" alt="image" src="https://github.com/user-attachments/assets/60a9a98b-d940-4920-93fa-272a2f81d06c" />
   问题3.如果现在给的文档不止是PDF，还有html的话怎么办？
   <img width="864" height="384" alt="image" src="https://github.com/user-attachments/assets/5ad7387a-5f2c-441f-9fae-6cd1babce8bb" />
   问题4.这个overlap的值是怎么调的，而且是怎么评判结果的准确性的？
   <img width="860" height="270" alt="image" src="https://github.com/user-attachments/assets/75a294ce-dda7-45c4-b76c-a4bb88ca3172" />
   问题5.如果数据有目录等没有用的内容，我在加载前是不是应该先对它进行处理？
   <img width="860" height="275" alt="image" src="https://github.com/user-attachments/assets/3a433609-d809-4199-bf52-084de389a735" />
   问题6.怎么确定现在的切分策略比较好？
   <img width="456" height="670" alt="image" src="https://github.com/user-attachments/assets/4a04e317-44b5-4c14-8510-03825e8a5a13" />
## DAY2
  1.建立索引库和BM25关键词库
  对于2份PDF是可以进行embedding的，但是对于多份就不行。->进行了增量版的建库（每次建库之前要删除原来的chroma旧库）
  
  <img width="864" height="250" alt="image" src="https://github.com/user-attachments/assets/7c8d9a02-de0a-46c6-a45a-87ad55893637" />
  
  问题1.切片时候的 overlap 后续是怎么用的？
  在切分那一刻就已经 "复制" 进相邻两个 chunk 的文本里了，从此跟着 chunk 走完全程
  
  <img width="550" height="711" alt="image" src="https://github.com/user-attachments/assets/125ea6cd-dc87-4820-b5f4-b7155e5f73e3" />
  
  问题2.这里向量库为啥要用chroma?
  
<img width="1016" height="498" alt="image" src="https://github.com/user-attachments/assets/595de6b3-8c3b-43d5-b272-2a299dd8aa97" />

   问题3.为啥大小 chunk 都建立了 overlap？
   
<img width="519" height="214" alt="image" src="https://github.com/user-attachments/assets/05b2ed50-b48c-4100-91af-b24b817d871c" />

   问题4.overlap会在后面建立索引吗，怎么进入后续流程的？
   
   <img width="825" height="416" alt="image" src="https://github.com/user-attachments/assets/b09447e3-0059-4111-94bb-4be7a28a603a" />
   
   问题5.BM25是怎么做的？BM25库里面存的数据是什么样的？
   
   <img width="845" height="95" alt="image" src="https://github.com/user-attachments/assets/86165fec-cb70-4a71-8ec9-4f7f0bf362c6" />

   <img width="775" height="205" alt="image" src="https://github.com/user-attachments/assets/4b339895-e595-4ac9-85e1-bc8e0089de4f" />

   问题6.BM25建的库放在哪儿，他和向量索引库合并吗？
   
   <img width="650" height="265" alt="image" src="https://github.com/user-attachments/assets/180ee876-02e1-4a25-acc2-880681959877" />

   问题7.之前embedding5个PDF文件会出错，现在修改了：
   <img width="761" height="185" alt="image" src="https://github.com/user-attachments/assets/961cc564-be6e-444f-b7a3-03d112872479" />
   <img width="890" height="119" alt="image" src="https://github.com/user-attachments/assets/5cbdfc1d-6109-420b-bc2a-71b2e75c3bb8" />
   
   2. 实现混合检索：向量召回负责语义相似，BM25 负责关键词精确，取并集后通过 parent_idx 映射到父块，返回完整上下文给生成阶段
      问题1.这里的top_k是怎么确定的？
      <img width="890" height="145" alt="image" src="https://github.com/user-attachments/assets/613ded33-75b5-4156-82c6-cb26689f0e90" />

      
   






   
