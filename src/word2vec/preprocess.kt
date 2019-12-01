package word2vec

/**
 * 指定された文字列から、以下の三つのオブジェクトを作成する。
 * ・単語→IDの連想配列
 * ・ID→単語の連想配列
 * ・指定された文字列を、単語→IDの連想配列を用いてInt配列に変換したもの
 */
//fun preprocess(text: String) {
//    val words = text
//            .toLowerCase()
//            .replace(".", " .")
//            .split(' ')
//
//    val word_to_id = HashMap<String, Int>()
//    val id_to_word = HashMap<Int, String>()
//    words.forEach { word ->
//        if (!word_to_id.contains(word)) {
//            val new_id = word_to_id.count()
//            word_to_id[word] = new_id
//            id_to_word[new_id] = word
//        }
//    }
//
//    corpus =  np.array([word_to_id[w] for w in words])
//
//    return corpus, word_to_id, id_to_word
//}
