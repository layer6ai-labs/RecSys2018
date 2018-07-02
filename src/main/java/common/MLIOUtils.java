package common;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class MLIOUtils {

	public static <T extends Serializable> T readObjectFromFile(
			final String file, Class<T> classType) throws Exception {
		if ((new File(file)).exists() == false) {
			throw new Exception("file doesn't exists " + file);
		}

		ObjectInputStream objectInputStream = null;
		try {
			BufferedInputStream fileInputStream = new BufferedInputStream(
					new FileInputStream(file));

			objectInputStream = new ObjectInputStream(fileInputStream);
			Object o = objectInputStream.readObject();

			if (o.getClass().equals(classType) == true) {
				return classType.cast(o);
			} else {
				throw new Exception("failed to deserialize " + file
						+ " inti class " + classType.getSimpleName());
			}

		} finally {
			if (objectInputStream != null) {
				objectInputStream.close();
			}
		}
	}

	public static <T extends Serializable> T readObjectFromFileGZ(
			final String file, Class<T> classType) throws Exception {
		if ((new File(file)).exists() == false) {
			throw new Exception("file doesn't exists " + file);
		}

		ObjectInputStream objectInputStream = null;
		try {
			BufferedInputStream fileInputStream = new BufferedInputStream(
					new FileInputStream(file));

			GZIPInputStream gzInputStream = new GZIPInputStream(
					fileInputStream);

			objectInputStream = new ObjectInputStream(gzInputStream);
			Object o = objectInputStream.readObject();

			if (o.getClass().equals(classType) == true) {
				return classType.cast(o);
			} else {
				throw new Exception("failed to serialize " + file
						+ " inti class " + classType.getSimpleName());
			}

		} finally {
			if (objectInputStream != null) {
				objectInputStream.close();
			}
		}
	}

	public static void writeObjectToFile(final Object object, final String file)
			throws IOException {
		ObjectOutputStream objectOutputStream = null;
		try {
			BufferedOutputStream fileOutputStream = new BufferedOutputStream(
					new FileOutputStream(file));

			objectOutputStream = new ObjectOutputStream(fileOutputStream);
			objectOutputStream.writeObject(object);

		} finally {
			if (objectOutputStream != null) {
				objectOutputStream.close();
			}
		}
	}

	public static void writeObjectToFileGZ(final Object object,
			final String file) throws IOException {
		ObjectOutputStream objectOutputStream = null;
		try {
			BufferedOutputStream fileOutputStream = new BufferedOutputStream(
					new FileOutputStream(file));

			GZIPOutputStream gzOutputStream = new GZIPOutputStream(
					fileOutputStream);

			objectOutputStream = new ObjectOutputStream(gzOutputStream);
			objectOutputStream.writeObject(object);

		} finally {
			if (objectOutputStream != null) {
				objectOutputStream.close();
			}
		}
	}

}
