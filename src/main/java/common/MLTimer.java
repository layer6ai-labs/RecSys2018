package common;

import java.util.concurrent.TimeUnit;

import com.google.common.base.Stopwatch;

public class MLTimer {

	private String name;
	private long loopSize;
	private Stopwatch timer;

	public MLTimer(final String nameP) {
		this.name = nameP;
		this.loopSize = 0;
		this.timer = Stopwatch.createUnstarted();
	}

	public MLTimer(final String nameP, final int loopSizeP) {
		this.name = nameP;
		this.loopSize = loopSizeP;
		this.timer = Stopwatch.createUnstarted();
	}

	public synchronized void tic() {
		this.timer.reset().start();
	}

	public synchronized void toc() {
		double elapsedTime = this.timer.elapsed(TimeUnit.MILLISECONDS) / 1000.0;
		System.out.printf("%s: elapsed [%s]\n", this.name,
				formatSeconds((float) elapsedTime));
	}

	public synchronized void toc(final String message) {
		double elapsedTime = this.timer.elapsed(TimeUnit.MILLISECONDS) / 1000.0;
		System.out.printf("%s: %s elapsed [%s]\n", this.name, message,
				formatSeconds((float) elapsedTime));
	}

	public synchronized void tocLoop(final int curLoop) {
		tocLoop("", curLoop);
	}

	public synchronized void tocLoop(String message, final int curLoop) {

		double elapsedTime = this.timer.elapsed(TimeUnit.MILLISECONDS) / 1000.0;
		double speed = curLoop / elapsedTime;

		if (this.loopSize > 0) {
			double remainTime = (this.loopSize - curLoop) / speed;

			System.out.printf(
					"%s: %s[%2.2f%%] elapsed [%s] cur_spd [%.0f/s] remain [%s]\n",
					this.name, message, (curLoop * 100f) / this.loopSize,
					formatSeconds(elapsedTime), speed,
					formatSeconds(remainTime));
		} else {
			System.out.printf("%s: %s [%d] elapsed [%s] cur_spd [%.0f/s]\n",
					this.name, message, curLoop, formatSeconds(elapsedTime),
					speed);
		}
	}

	private static String formatSeconds(double secondsF) {
		if (secondsF < 0) {
			return Double.toString(secondsF);
		}
		TimeUnit base = TimeUnit.SECONDS;
		int s = (int) Math.floor(secondsF);
		// float remainder = (float) (secondsF - s);

		long days = base.toDays(s);
		s -= TimeUnit.DAYS.toSeconds(days);
		long hours = base.toHours(s);
		s -= TimeUnit.HOURS.toSeconds(hours);
		long minutes = base.toMinutes(s);
		s -= TimeUnit.MINUTES.toSeconds(minutes);
		long secondsL = base.toSeconds(s);

		StringBuilder sb = new StringBuilder();
		if (days > 0) {
			sb.append(days);
			sb.append(" days ");
		}
		if (hours > 0 || days > 0) {
			sb.append(hours);
			sb.append(" hr ");
		}
		if (hours > 0 || days > 0 || minutes > 0) {
			sb.append(minutes);
			sb.append(" min ");
		}
		sb.append(secondsL + " sec");

		return sb.toString();
	}

}
