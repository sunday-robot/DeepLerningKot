package word2vec

/**
 * コンテキストとターゲットのリストを作成する
 * @param corpus 単語IDのリスト
 * @param windowSize コンテキストのサイズ(1ならターゲットの単語の前後１単語をコンテキストとし、2なら前後の２単語をコンテキストとする)
 */
fun createTargetAndContextList(corpus: List<Int>, windowSize: Int = 1) =
        List(corpus.size - windowSize * 2) { i ->
            val idx = i + windowSize
            val context = arrayListOf<Int>()

            // 手前の単語を登録する
            for (t in (idx - windowSize).until(idx))
                context.add(corpus[t])

            // 後続の単語を登録する
            for (t in (idx + 1).until(idx + 1 + windowSize))
                context.add(corpus[t])

            TargetAndContext(corpus[idx], context)
        }
