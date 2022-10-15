# twitter-preprocessing 

Twitter'dan snscrape aracılığıyla veri almak ve onu veri önişleme süreçlerinden geçirmek için bir tool. 

Direkt olarak tweetleri alıp bir CSV dosyası oluşturabilir ya da işlemek üzere bir dataframe'e kaydedebilirsiniz.

Arama yapmak için **[Twitter Advanced Search]**'ü kullanabilirsiniz. Buradan alacağınız query'i doğrudan uygulamaya ekleyebilirsiniz. 

<p align="center">
  <img src="./img/advanced.png" alt="Twitter Advanced Search" width="500">
</p>

Buradan elde ettiğimiz query'i direkt olarak uygulamamıza verebiliriz.

```python
query = "(zelenski OR zelensky OR zelenskiy) lang:tr"
```

Verimizi elde etmek ve daha sonrasında kullanmak için, `csv_only` değerini `True` olarak belirleyebiliriz.

Daha sonra kullanmak üzere CSV olarak kaydetmek için.

```python
get_tweets(query, 100, csv_only=True)
```

Direkt olarak üstünde çalışmak için.

```python
tweets = get_tweets(query, 100)
```

Özellikle `transformers` kütüphanesi `CUDA` ve `PyTorch` gerektirdiğinden ilk çalıştırma için sorun oluşturabilir. Bu sebepten ötürü direkt olarak çalışmaya ait **[Google Colab Versiyonu]** kullanabilirsiniz.

[Twitter Advanced Search]: https://twitter.com/search-advanced
[Google Colab Versiyonu]: https://colab.research.google.com/drive/17BZVWFrTTd1_UC42Q9Iigg1e5K3ieyWS?usp=sharing
