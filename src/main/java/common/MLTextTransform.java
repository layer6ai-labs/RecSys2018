package common;

import java.io.IOException;
import java.io.Serializable;
import java.io.StringReader;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.miscellaneous.ASCIIFoldingFilter;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.miscellaneous.WordDelimiterGraphFilterFactory;
import org.apache.lucene.analysis.ngram.NGramTokenFilter;

public abstract class MLTextTransform implements Serializable {

	public static class DefaultAnalyzer extends Analyzer
			implements Serializable {
		private static final long serialVersionUID = 8662472589510750438L;

		@Override
		protected TokenStreamComponents createComponents(
				final String fieldName) {
			try {
				Tokenizer tokenizer = new WhitespaceTokenizer();

				Map<String, String> args = new HashMap<String, String>();
				args.put("catenateAll", "0");
				args.put("generateNumberParts", "1");
				args.put("generateWordParts", "1");
				args.put("splitOnCaseChange", "1");
				args.put("splitOnNumerics", "1");

				WordDelimiterGraphFilterFactory factory = new WordDelimiterGraphFilterFactory(
						args);
				TokenStream filter = factory.create(tokenizer);

				filter = new LowerCaseFilter(filter);

				filter = new ASCIIFoldingFilter(filter);

				return new TokenStreamComponents(tokenizer, filter);

			} catch (Exception e) {
				e.printStackTrace();
			}
			return null;
		}
	}

	public static class DefaultNGRAMAnalyzer extends Analyzer
			implements Serializable {

		private static final long serialVersionUID = 2224016723762685329L;
		// https://lucene.apache.org/core/7_0_0/analyzers-common/org/apache/lucene/analysis/ngram/NGramTokenFilter.html
		private int minGram;
		private int maxGram;

		public DefaultNGRAMAnalyzer(final int minGraP, final int maxGramP) {
			this.minGram = minGraP;
			this.maxGram = maxGramP;
		}

		@Override
		protected TokenStreamComponents createComponents(
				final String fieldName) {
			try {
				Tokenizer tokenizer = new WhitespaceTokenizer();

				Map<String, String> args = new HashMap<String, String>();
				args.put("catenateAll", "0");
				args.put("generateNumberParts", "1");
				args.put("generateWordParts", "1");
				args.put("splitOnCaseChange", "1");
				args.put("splitOnNumerics", "1");

				WordDelimiterGraphFilterFactory factory = new WordDelimiterGraphFilterFactory(
						args);
				TokenStream filter = factory.create(tokenizer);

				filter = new LowerCaseFilter(filter);

				filter = new ASCIIFoldingFilter(filter);

				filter = new NGramTokenFilter(filter, this.minGram,
						this.maxGram);

				return new TokenStreamComponents(tokenizer, filter);

			} catch (Exception e) {
				e.printStackTrace();
			}
			return null;
		}
	}

	public static class LuceneAnalyzerTextTransform extends MLTextTransform {

		private static final long serialVersionUID = 1843607513745972795L;
		private Analyzer analyzer;

		public LuceneAnalyzerTextTransform(final Analyzer analyzerP) {
			this.analyzer = analyzerP;
		}

		@Override
		public void apply(final MLTextInput input) {
			try {
				List<String> tokens = passThroughAnalyzer(input.text,
						this.analyzer);
				String[] tokenized = new String[tokens.size()];
				int cur = 0;
				for (String token : tokens) {
					tokenized[cur] = token;
					cur++;
				}
				input.setTokenized(tokenized);

			} catch (Exception e) {
				throw new RuntimeException(e.getMessage());
			}
		}

		public static List<String> passThroughAnalyzer(final String input,
				final Analyzer analyzer) throws IOException {
			TokenStream tokenStream = null;
			try {
				tokenStream = analyzer.tokenStream(null,
						new StringReader(input));
				CharTermAttribute termAtt = tokenStream
						.addAttribute(CharTermAttribute.class);
				tokenStream.reset();
				List<String> tokens = new LinkedList<String>();
				while (tokenStream.incrementToken()) {
					String term = termAtt.toString().trim();
					if (term.length() > 0) {
						tokens.add(term);
					}
				}
				tokenStream.end();

				return tokens;
			} finally {
				if (tokenStream != null) {
					tokenStream.close();
				}
			}
		}
	}

	public static class MLTextInput {

		private String text;
		private String[] tokenized;

		public MLTextInput(final String textP) {
			this.text = textP;
		}

		public String getText() {
			return this.text;
		}

		public String[] getTokenized() {
			return this.tokenized;
		}

		public void setTokenized(final String[] tokenizedP) {
			this.tokenized = tokenizedP;
		}
	}

	private static final long serialVersionUID = 3800020927323228525L;

	public abstract void apply(final MLTextInput input);
}
