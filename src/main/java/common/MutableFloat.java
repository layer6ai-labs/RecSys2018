package common;

import java.io.Serializable;

public class MutableFloat implements Serializable {

	private static final long serialVersionUID = -3705775132945867924L;
	public float value;

	public MutableFloat(final float valueP) {
		this.value = valueP;
	}
}
