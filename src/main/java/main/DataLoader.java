package main;

import net.minidev.json.JSONArray;
import net.minidev.json.JSONObject;
import net.minidev.json.JSONValue;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import common.MLTimer;

public class DataLoader {

	public static Data load(final String trainPath, final String testFileName)
			throws IOException {

		File folder = new File(trainPath);
		File[] listOfFiles = folder.listFiles();
		Arrays.sort(listOfFiles);
		File testFile = new File(testFileName);

		Map<String, Integer> songIdToIndex = new HashMap<String, Integer>();

		AtomicInteger uniquePlaylistCounter = new AtomicInteger(0);
		AtomicInteger uniqueSongCounter = new AtomicInteger(0);
		AtomicInteger parsedSongCounter = new AtomicInteger(0);

		List<Playlist> playlists = new ArrayList<Playlist>();
		List<Song> songs = new ArrayList<Song>();
		List<Integer> testIndexes = new ArrayList<Integer>();
		MLTimer timer = new MLTimer("load");
		timer.tic();

		for (int f = 0; f <= listOfFiles.length; f++) {
			File file;
			if (f >= listOfFiles.length) {
				timer.toc("test file " + testFileName);
				file = testFile;
			} else {
				file = listOfFiles[f];
			}

			try (BufferedReader reader = new BufferedReader(
					new FileReader(file))) {
				JSONObject obj = (JSONObject) JSONValue.parse(reader);
				JSONArray list = (JSONArray) obj.get("playlists");
				for (int l = 0; l < list.size(); l++) {
					uniquePlaylistCounter.incrementAndGet();
					if (f >= listOfFiles.length) {
						testIndexes.add(uniquePlaylistCounter.get() - 1);
					}

					Object data = list.get(l);
					Playlist playlist = new Playlist((JSONObject) data);

					Object tracksObj = ((JSONObject) data).get("tracks");
					if (tracksObj != null && tracksObj instanceof JSONArray) {
						JSONArray tracksArray = (JSONArray) tracksObj;
						Track[] tracks = new Track[tracksArray.size()];
						for (int i = 0; i < tracksArray.size(); i++) {
							JSONObject songObj = (JSONObject) tracksArray
									.get(i);
							Song song = new Song(songObj);
							Integer songIndex = songIdToIndex
									.get(song.get_track_uri());
							if (songIndex == null) {
								songIndex = uniqueSongCounter.getAndIncrement();
								songIdToIndex.put(song.get_track_uri(),
										songIndex);
								songs.add(song);
							}
							tracks[i] = new Track(songIndex,
									songObj.getAsNumber("pos").intValue());
							parsedSongCounter.incrementAndGet();
						}
						playlist.setTracks(tracks);
					}
					playlists.add(playlist);
				}

				if ((f + 1) % 10 == 0) {
					timer.tocLoop(String.format(
							"playlists[%d] unique songs[%d] total songs[%d]",
							playlists.size(), songs.size(),
							parsedSongCounter.get()), parsedSongCounter.get());
				}
			}
		}

		System.out.printf(
				"FINISHED PARSING: playlists[%d] unique songs[%d] total songs[%d]",
				playlists.size(), songs.size(), parsedSongCounter.get());
		Data data = new Data();

		data.playlists = new Playlist[playlists.size()];
		playlists.toArray(data.playlists);

		data.testIndexes = testIndexes.stream().mapToInt(i -> i).toArray();

		data.songs = new Song[songs.size()];
		songs.toArray(data.songs);

		return data;

	}

}
