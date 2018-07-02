package common;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import common.MLTextTransform.MLTextInput;

public class MLSparseFeature implements Serializable {

    private static final long serialVersionUID = 4665530401588164620L;

    private MLSparseMatrix featMatrix;
    private MLSparseMatrix featMatrixTrans;
    private Map<String, Integer> catToIndex;
    private Map<Integer, String> indexToCat;
    private AtomicInteger curCatIndex;
    // NOTE: last transform in this sequence must tokenize text
    private MLTextTransform[] textTransforms;
    private MLFeatureTransform[] featTransforms;
    private AtomicBoolean inInfMode;
    private int nRows;
    private Class<? extends MLSparseMatrix> type;

    private transient MLSparseMatrix featMatrixCache;
    private transient MLSparseMatrix featMatrixTransCache;

    public <T extends MLSparseMatrix> MLSparseFeature(final int nRowsP,
                                                      final MLTextTransform[] textTransformsP,
                                                      final MLFeatureTransform[] featTransformsP, final Class<T> typeP) {
        this.nRows = nRowsP;
        this.type = typeP;

        this.catToIndex = new HashMap<String, Integer>();
        this.indexToCat = new HashMap<Integer, String>();
        this.curCatIndex = new AtomicInteger(-1);

        this.textTransforms = textTransformsP;
        this.featTransforms = featTransformsP;
        this.inInfMode = new AtomicBoolean(false);

        this.prepareForData();
    }

    public <T extends MLSparseMatrix> MLSparseFeature(final int nRowsP,
                                                      final MLTextTransform[] textTransformsP,
                                                      final MLFeatureTransform[] transformsP, final Class<T> typeP,
                                                      final MLSparseFeature anotherFeature) {
        this(nRowsP, textTransformsP, transformsP, typeP);

        // share category maps with another feature
        this.indexToCat = anotherFeature.indexToCat;
        this.catToIndex = anotherFeature.catToIndex;
        this.curCatIndex = anotherFeature.curCatIndex;
    }

    public <T extends MLSparseMatrix> MLSparseFeature(
            final MLFeatureTransform[] transformsP,
            final MLSparseMatrix featMatrixP) {
        this(featMatrixP.getNRows(), null, transformsP, featMatrixP.getClass());

        // init feature with existing data matrix
        this.featMatrix = featMatrixP;
    }

    public void addRow(final int rowIndex, final float value) {
        if (this.type.equals(MLSparseMatrixFlat.class) == true) {
            ((MLSparseMatrixFlat) this.featMatrix).setRow(0, value, rowIndex);

        } else {
            this.featMatrix.setRow(new MLSparseVector(new int[]{0},
                    new float[]{value}, null, 1), rowIndex);
        }
    }

    public void addRow(final int rowIndex, final float value, final int index) {
        if (this.type.equals(MLSparseMatrixFlat.class) == true) {
            ((MLSparseMatrixFlat) this.featMatrix).setRow(index, value,
                    rowIndex);

        } else {
            this.featMatrix.setRow(new MLSparseVector(new int[]{index},
                    new float[]{value}, null, 1), rowIndex);
        }
    }

    public void addRow(final int rowIndex, final MLDenseVector dense) {
        MLSparseVector sparse = dense.toSparse();
        if (sparse.getIndexes() != null) {
            this.featMatrix.setRow(sparse, rowIndex);
        } else {
            this.featMatrix.setRow(null, rowIndex);
        }
    }

    public void addRow(final int rowIndex, final MLSparseVector sparse) {
        if (sparse.getIndexes() != null) {
            this.featMatrix.setRow(sparse, rowIndex);
        } else {
            this.featMatrix.setRow(null, rowIndex);
        }
    }

    public void addRow(final int rowIndex, final String text) {

        if (this.textTransforms == null) {
            // no transforms so treat as category
            Integer index = this.getCatIndex(text);
            if (index == null) {
                this.featMatrix.setRow(null, rowIndex);
                return;
            }

            if (this.type.equals(MLSparseMatrixFlat.class) == true) {
                // can set row directly here
                ((MLSparseMatrixFlat) this.featMatrix).setRow(index, 1,
                        rowIndex);
            } else {
                this.featMatrix.setRow(new MLSparseVector(new int[]{index},
                                new float[]{1}, null, this.catToIndex.size()),
                        rowIndex);
            }

        } else {
            // apply transforms and tokenize
            MLTextInput input = new MLTextInput(text);
            for (MLTextTransform inputTransform : this.textTransforms) {
                inputTransform.apply(input);
            }

            String[] tokenized = input.getTokenized();
            if (tokenized != null && tokenized.length > 0) {
                this.addRow(rowIndex, tokenized);
            } else {
                this.featMatrix.setRow(null, rowIndex);
            }
        }
    }

    public void addRow(final int rowIndex, final String[] cats) {
        // map values to indexes and add sparse row to matrix
        MLSparseVector sparse = this.getFeatVector(cats);
        if (sparse.getIndexes() != null) {
            this.featMatrix.setRow(sparse, rowIndex);
        } else {
            this.featMatrix.setRow(null, rowIndex);
        }
    }

    public void finalizeFeature(final boolean preserveOrig) {
        // NOTE: this fn must be called before feature can be used

        // infer nCols
        this.featMatrix.inferAndSetNCols();

        // this is necessary for features with shared cat maps
        int nColsCat = 0;
        for (Integer index : this.catToIndex.values()) {
            if (nColsCat < (index + 1)) {
                nColsCat = index + 1;
            }
        }
        if (this.featMatrix.getNCols() < nColsCat) {
            this.featMatrix.setNCols(nColsCat);
        }

        // apply all transforms
        if (preserveOrig == true) {
            // deep copy
            this.featMatrixTrans = this.featMatrix.deepCopy();
        } else {
            // shallow copy
            this.featMatrixTrans = this.featMatrix;
        }

        if (this.featTransforms != null) {
            for (MLFeatureTransform transform : this.featTransforms) {
                if (this.inInfMode.get() == true) {
                    // don't recalculate transforms in inf mode
                    transform.applyInference(this);
                } else {
                    transform.apply(this);
                }
            }
        }
        this.inInfMode.set(true);
    }

    public void finishSerialize() {
        // NOTE: must call this after serialization
        if (this.featMatrix == null) {
            this.featMatrix = this.featMatrixCache;
            this.featMatrixTrans = this.featMatrixTransCache;
        }
    }


    private synchronized Integer getCatIndex(final String cat) {
        Integer index = this.catToIndex.get(cat);
        if (index == null) {
            if (this.inInfMode.get() == false) {
                index = this.curCatIndex.incrementAndGet();

                this.catToIndex.put(cat, index);
                this.indexToCat.put(index, cat);
            }
        }
        return index;
    }

    public Map<String, Integer> getCatToIndex() {
        return this.catToIndex;
    }

    public MLSparseVector getFeatInf(final float value) {
        MLSparseVector vector = new MLSparseVector(new int[]{0},
                new float[]{value}, null, 1);

        // apply feature transforms
        if (this.featTransforms != null) {
            for (MLFeatureTransform transform : this.featTransforms) {
                transform.applyInference(vector);
            }
        }
        return vector;
    }

    public MLSparseVector getFeatInf(final MLDenseVector dense) {
        if (this.inInfMode.get() == false) {
            throw new IllegalStateException(
                    "feature is not in inference mode, call finalizeFeature()");
        }

        if (this.indexToCat.size() > 0
                && dense.getLength() != this.indexToCat.size()) {
            throw new IllegalArgumentException(
                    "dense.getLength() != this.indexToCat.size()");
        }

        // map dense to sparse
        MLSparseVector vector = dense.toSparse();

        // apply feature transforms
        if (this.featTransforms != null) {
            for (MLFeatureTransform transform : this.featTransforms) {
                transform.applyInference(vector);
            }
        }
        return vector;
    }

    public MLSparseVector getFeatInf(final MLSparseVector sparse) {
        if (this.inInfMode.get() == false) {
            throw new IllegalStateException(
                    "feature is not in inference mode, call finalizeFeature()");
        }

        if (this.indexToCat.size() > 0
                && sparse.getLength() != this.indexToCat.size()) {
            throw new IllegalArgumentException(
                    "dense.getLength() != this.indexToCat.size()");
        }

        // apply feature transforms
        if (this.featTransforms != null) {
            for (MLFeatureTransform transform : this.featTransforms) {
                transform.applyInference(sparse);
            }
        }
        return sparse;
    }

    public MLSparseVector getFeatInf(final String text) {
        if (this.inInfMode.get() == false) {
            throw new IllegalStateException(
                    "feature is not in inference mode, call finalizeFeature()");
        }

        MLSparseVector vector = null;
        if (this.textTransforms == null) {
            // no text transforms so treat this as category
            vector = this.getFeatVector(text);
        } else {

            // apply text transforms
            MLTextInput input = new MLTextInput(text);
            for (MLTextTransform inputTransform : this.textTransforms) {
                inputTransform.apply(input);
            }

            // map tokenized text to sparse vector
            String[] tokenized = input.getTokenized();
            vector = this.getFeatVector(tokenized);
        }

        // apply feature transforms
        if (this.featTransforms != null) {
            for (MLFeatureTransform transform : this.featTransforms) {
                transform.applyInference(vector);
            }
        }
        return vector;
    }

    public MLSparseVector getFeatInf(final String[] cats) {
        if (this.inInfMode.get() == false) {
            throw new IllegalStateException(
                    "feature is not in inference mode, call finalizeFeature()");
        }

        // map categories to sparse
        MLSparseVector vector = this.getFeatVector(cats);

        // apply all the transforms
        if (this.featTransforms != null) {
            for (MLFeatureTransform transform : this.featTransforms) {
                transform.applyInference(vector);
            }
        }
        return vector;
    }

    public MLSparseMatrix getFeatMatrix() {
        return this.featMatrix;
    }

    public MLSparseMatrix getFeatMatrixTransformed() {
        return this.featMatrixTrans;
    }

    public String[] getFeatNames(final String prefix,
                                 final boolean transformed) {
        if (this.indexToCat.size() == 0) {
            // numerical feature
            String[] featNames = new String[]{prefix};
            if (transformed == true) {
                // get feature names after all transforms are applied
                for (MLFeatureTransform transform : this.featTransforms) {
                    featNames = transform.applyFeatureName(featNames);
                }
            }
            return featNames;
        }
        int[] keys = new int[this.indexToCat.size()];
        int cur = 0;
        for (int key : this.indexToCat.keySet()) {
            keys[cur] = key;
            cur++;
        }
        Arrays.sort(keys);

        // get feature name in the format 'prefix_[cat name]'
        String[] featNames = new String[keys.length];
        for (int i = 0; i < keys.length; i++) {
            featNames[i] = prefix + "_" + this.indexToCat.get(keys[i]).trim()
                    .replaceAll("\\s+", "_");
        }

        if (transformed == true) {
            // get feature names after all transforms are applied
            for (MLFeatureTransform transform : this.featTransforms) {
                featNames = transform.applyFeatureName(featNames);
            }
        }

        return featNames;
    }

    private MLSparseVector getFeatVector(final String cat) {
        Integer index = this.getCatIndex(cat);
        if (index == null) {
            return new MLSparseVector(null, null, null, this.catToIndex.size());
        } else {
            return new MLSparseVector(new int[]{index}, new float[]{1},
                    null, this.catToIndex.size());
        }
    }

    private MLSparseVector getFeatVector(final String[] cats) {
        TreeMap<Integer, MutableFloat> countMap = new TreeMap<Integer, MutableFloat>();
        for (int i = 0; i < cats.length; i++) {
            Integer index = this.getCatIndex(cats[i]);
            if (index == null) {
                continue;
            }

            MutableFloat count = countMap.get(index);
            if (count == null) {
                countMap.put(index, new MutableFloat(1f));

            } else {
                count.value++;
            }
        }

        if (countMap.size() == 0) {
            new MLSparseVector(null, null, null, this.catToIndex.size());
        }

        int[] indexes = new int[countMap.size()];
        float[] values = new float[countMap.size()];
        int cur = 0;
        for (Map.Entry<Integer, MutableFloat> entry : countMap.entrySet()) {
            indexes[cur] = entry.getKey();
            values[cur] = entry.getValue().value;
            cur++;
        }

        // NOTE catToIndex.size() can be wrong in non-inference
        // mode since we don't know number of categories ahead of
        // time. This is corrected by finalizeFeature().
        return new MLSparseVector(indexes, values, null,
                this.catToIndex.size());
    }

    public Map<Integer, String> getIndexToCat() {
        return this.indexToCat;
    }

    public MLSparseVector getRow(final int rowIndex,
                                 final boolean returnEmpty) {
        return this.featMatrix.getRow(rowIndex, returnEmpty);
    }

    public MLSparseVector getRowTransformed(final int rowIndex,
                                            final boolean returnEmpty) {
        return this.featMatrixTrans.getRow(rowIndex, returnEmpty);
    }

    public boolean infMode() {
        return this.inInfMode.get();
    }

    public void prepareForData(final int nRowsP) {
        if (this.featMatrix == null) {
            this.nRows = nRowsP;
            if (this.type.equals(MLSparseMatrixAOO.class) == true) {
                this.featMatrix = new MLSparseMatrixAOO(this.nRows, 0);

            } else if (this.type.equals(MLSparseMatrixFlat.class) == true) {
                this.featMatrix = new MLSparseMatrixFlat(this.nRows, 0);

            } else {
                throw new IllegalArgumentException(
                        "unsupported type " + this.type.getName());
            }
        }
    }

    public void prepareForData() {
        prepareForData(this.nRows);
    }

    public void prepareToSerialize(final boolean withData) {
        if (withData == false) {
            // NOTE: this excludes data from serialization so
            // all data loading must be re-run and you must call
            // finalize before using this feature after de-serialization.
            this.featMatrixCache = this.featMatrix;
            this.featMatrixTransCache = this.featMatrixTrans;

            this.featMatrix = null;
            this.featMatrixTrans = null;
            // NOTE: must call finishSerialize() if the data in this
            // feature is to be used after serialization.
        }
    }
}
