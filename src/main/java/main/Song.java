package main;

import net.minidev.json.JSONObject;

import java.io.Serializable;

public class Song implements Serializable {

	private static final long serialVersionUID = 6265625137029257218L;
	private String artist_name;
	private String track_uri;
	private String artist_uri;
	private String track_name;
	private String album_uri;
	private int duration_ms;
	private String album_name;

	public Song(final JSONObject obj) {
		this.artist_name = obj.getAsString("artist_name");
		this.track_uri = obj.getAsString("track_uri");
		this.artist_uri = obj.getAsString("artist_uri");
		this.track_name = obj.getAsString("track_name");
		this.album_uri = obj.getAsString("album_uri");
		this.duration_ms = obj.getAsNumber("duration_ms").intValue();
		this.album_name = obj.getAsString("album_name");
	}

	public String get_artist_name() {
		return this.artist_name;
	}

	public String get_track_uri() {
		return this.track_uri;
	}

	public String get_artist_uri() {
		return this.artist_uri;
	}

	public String get_track_name() {
		return this.track_name;
	}

	public String get_album_uri() {
		return this.album_uri;
	}

	public int get_duration_ms() {
		return this.duration_ms;
	}

	public String get_album_name() {
		return this.album_name;
	}

}
