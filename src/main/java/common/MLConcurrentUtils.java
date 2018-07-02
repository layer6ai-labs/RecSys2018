package common;

import java.util.HashMap;
import java.util.Iterator;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.*;

public class MLConcurrentUtils {
	public static class Async<T> extends ThreadLocal<T> {
		private HashMap<Integer, T> refs;
		private AtomicInteger threadCounter;

		private final Supplier<T> constructor;
		private final Consumer<T> cleaner;

		public Async(Supplier<T> constructorP, Consumer<T> cleanerP) {
			this.refs = new HashMap<Integer, T>();
			this.threadCounter = new AtomicInteger(0);
			this.constructor = constructorP;
			this.cleaner = cleanerP;
		}

		public void cleanAll() {
			if (this.cleaner != null) {
				synchronized (this.refs) {
					this.refs.values().forEach(this.cleaner::accept);
				}
			}
		}

		@Override protected T initialValue() {
			int localCount = this.threadCounter.getAndIncrement();
			T t = this.constructor.get();
			this.refs.put(localCount, t);
			return t;
		}
	}

	public static class PreloadingQueue<T> implements AutoCloseable {
		private static class QUpdater<U> implements Runnable {

			private Iterator<U> src;
			private ArrayBlockingQueue<U> raw;

			private QUpdater(Iterator<U> srcP, ArrayBlockingQueue<U> rawP) {
				src = srcP;
				raw = rawP;
			}

			public void addOneNow() {
				if (raw.remainingCapacity() > 0 && src.hasNext()) {
					raw.add(src.next());
				}
			}

			public boolean hasMore() {
				synchronized (src) {
					return src.hasNext();
				}
			}

			@Override public void run() {
				// check and fill queue
				while (raw.remainingCapacity() > 0 && src.hasNext()) {
					raw.add(src.next());
				}
			}
		}
		private ArrayBlockingQueue<T> raw;
		private QUpdater<T> updater;

		private ExecutorService updatePool;

		public PreloadingQueue(Iterator<T> src, int maxQueueSize) {
			this(src, maxQueueSize, Executors.newFixedThreadPool(1));
		}

		public PreloadingQueue(Iterator<T> src, int maxQueueSize,
				ExecutorService pool) {
			raw = new ArrayBlockingQueue<>(maxQueueSize);
			updater = new QUpdater<>(src, raw);
			updatePool = pool;
		}

		@Override public void close() throws Exception {
			if (this.updatePool != null) {
				this.updatePool.shutdownNow();
			}
		}

		public boolean hasMore() {
			return raw.isEmpty() == false || updater.hasMore();
		}

		public T pop() throws InterruptedException {
			T data = raw.take();
			requestUpdate();
			return data;
		}

		private void requestUpdate() {
			updatePool.submit(updater);
		}

		public void warmupOneBlocking() {
			updater.addOneNow();
		}

	}
}
