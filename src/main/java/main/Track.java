package main;

import java.io.Serializable;

public class Track implements Serializable {

	private static final long serialVersionUID = -4185883780088342841L;

	private int songIndex;
	private int songPos;

	public Track(final int songIndexP, final int songPosP) {
		this.songIndex = songIndexP;
		this.songPos = songPosP;
	}

	public int getSongIndex() {
		return this.songIndex;
	}

	public int getSongPos() {
		return this.songPos;
	}
}
