package main;

import common.MLDenseMatrix;
import main.ParsedData.PlaylistFeature;
import main.ParsedData.SongFeature;

public class Latents {

	public MLDenseMatrix U;
	public MLDenseMatrix V;

	public MLDenseMatrix Ucnn;
	public MLDenseMatrix Vcnn;

	public MLDenseMatrix Uname;
	public MLDenseMatrix Vname;
	public MLDenseMatrix name;

	public MLDenseMatrix Uartist;
	public MLDenseMatrix Vartist;
	public MLDenseMatrix artist;

	public MLDenseMatrix Ualbum;
	public MLDenseMatrix Valbum;
	public MLDenseMatrix album;

	public Latents() {

	}

	public Latents(final ParsedData data) throws Exception {
		String dataPath = "/media/mvolkovs/external4TB/Data/recsys2018";

		int rankWarm = 200;
		this.U = MLDenseMatrix
				.fromFile(
						dataPath + "/models/latent_song/matching_name_U_"
								+ rankWarm + ".bin",
						data.interactions.getNRows(), rankWarm);

		this.V = MLDenseMatrix
				.fromFile(
						dataPath + "/models/latent_song/matching_name_V_"
								+ rankWarm + ".bin",
						data.interactions.getNCols(), rankWarm);

		int rankWarmCNN = 200;
		this.Ucnn = MLDenseMatrix
				.fromFile(
						dataPath + "/models/latent_song/matching_CNN_v2_U_"
								+ rankWarm + ".bin",
						data.interactions.getNRows(), rankWarmCNN);

		this.Vcnn = MLDenseMatrix
				.fromFile(
						dataPath + "/models/latent_song/matching_CNN_v2_V_"
								+ rankWarm + ".bin",
						data.interactions.getNCols(), rankWarmCNN);

		int rankName = 200;
		this.Uname = MLDenseMatrix.fromFile(
				dataPath + "/models/latent_name/name_U_" + rankName + ".bin",
				data.interactions.getNRows(), rankName);
		this.Vname = MLDenseMatrix.fromFile(
				dataPath + "/models/latent_name/name_V_" + rankName + ".bin",
				data.interactions.getNCols(), rankName);
		this.name = MLDenseMatrix.fromFile(
				dataPath + "/models/latent_name/name_" + rankName + ".bin",
				data.playlistFeatures.get(PlaylistFeature.NAME_REGEXED)
						.getCatToIndex().size(),
				rankName);

		int rankArtist = 200;
		this.Uartist = MLDenseMatrix
				.fromFile(
						dataPath + "/models/latent_artist/artist_U_"
								+ rankArtist + ".bin",
						data.interactions.getNRows(), rankArtist);
		this.Vartist = MLDenseMatrix
				.fromFile(
						dataPath + "/models/latent_artist/artist_V_"
								+ rankArtist + ".bin",
						data.interactions.getNCols(), rankArtist);
		this.artist = MLDenseMatrix.fromFile(
				dataPath + "/models/latent_artist/artist_" + rankArtist
						+ ".bin",
				data.songFeatures.get(SongFeature.ARTIST_ID).getCatToIndex()
						.size(),
				rankArtist);

		int rankAlbum = 200;
		this.Ualbum = MLDenseMatrix.fromFile(
				dataPath + "/models/latent_album/album_U_" + rankAlbum + ".bin",
				data.interactions.getNRows(), rankAlbum);
		this.Valbum = MLDenseMatrix.fromFile(
				dataPath + "/models/latent_album/album_V_" + rankAlbum + ".bin",
				data.interactions.getNCols(), rankAlbum);
		this.album = MLDenseMatrix.fromFile(
				dataPath + "/models/latent_album/album_" + rankAlbum + ".bin",
				data.songFeatures.get(SongFeature.ALBUM_ID).getCatToIndex()
						.size(),
				rankAlbum);

	}
}
