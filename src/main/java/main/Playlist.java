package main;

import net.minidev.json.JSONObject;

import java.io.Serializable;

public class Playlist implements Serializable {

	private static final long serialVersionUID = 6071968443385525640L;
	public String name;
	public Boolean collaborative;
	public String pid;
	public Long modified_at;
	public Integer num_albums;
	public Integer num_tracks;
	public Integer num_followers;
	public Integer num_edits;
	public Integer duration_ms;
	public Integer num_artists;
	public Track[] tracks;

	public Playlist(final JSONObject obj) {
		this.pid = obj.getAsString("pid");
		this.name = obj.getAsString("name");

		if (obj.containsKey("collaborative") == true) {
			this.collaborative = obj.getAsString("collaborative").toLowerCase()
					.equals("true");
		}
		if (obj.containsKey("modified_at") == true) {
			this.modified_at = obj.getAsNumber("modified_at").longValue();
		}
		if (obj.containsKey("num_albums") == true) {
			this.num_albums = obj.getAsNumber("num_albums").intValue();
		}
		if (obj.containsKey("num_tracks") == true) {
			this.num_tracks = obj.getAsNumber("num_tracks").intValue();
		}
		if (obj.containsKey("num_followers") == true) {
			this.num_followers = obj.getAsNumber("num_followers").intValue();
		}
		if (obj.containsKey("num_edits") == true) {
			this.num_edits = obj.getAsNumber("num_edits").intValue();
		}
		if (obj.containsKey("duration_ms") == true) {
			this.duration_ms = obj.getAsNumber("duration_ms").intValue();
		}
		if (obj.containsKey("num_artists") == true) {
			this.num_artists = obj.getAsNumber("num_artists").intValue();
		}
	}

	public Boolean get_collaborative() {
		return this.collaborative;
	}

	public Integer get_duration_ms() {
		return this.duration_ms;
	}

	public Long get_modified_at() {
		return this.modified_at;
	}

	public String get_name() {
		return this.name;
	}

	public Integer get_num_albums() {
		return this.num_albums;
	}

	public Integer get_num_artists() {
		return this.num_artists;
	}

	public Integer get_num_edits() {
		return this.num_edits;
	}

	public Integer get_num_followers() {
		return this.num_followers;
	}

	public Integer get_num_tracks() {
		return this.num_tracks;
	}

	public String get_pid() {
		return this.pid;
	}

	public Track[] getTracks() {
		return this.tracks;
	}

	public void setTracks(final Track[] tracksP) {
		this.tracks = tracksP;
	}

}
