## DAY 1
   用langchain＋Olloma实现，对数据进行了加载，进行了文档切分为chunk处理，主要用到了：父子切割和语义切割，这里语义切分用到了embedding 模型（omic-embed-text）对句子之间进行相似度的计算，从而提高了切分的准确性。
   问题1.对于PDF文件中出现的表格处理不当，这里用到的pypdf会让表格乱序。
   <img width="819" height="355" alt="image" src="https://github.com/user-attachments/assets/1697bfb7-af8f-4012-a5ad-ce7444f52b82" /># rag-doc-qa
   问题2.这里embedding模型的选择会影响我的结果吗？
   问题3.如果现在给的文档不止是PDF，还有html的话怎么办？
   问题4.这个overlap的值是怎么调的，而且是怎么评判结果的准确性的？
   问题5.如果数据有目录等没有用的内容，我在加载前是不是应该先对它进行处理？

   
