(ns liu.mars.ml.repl
  (:require [clojure.string :as cs]
            [liu.mars.ml.neural :as n]))

(defn parse-int
  [item]
  (Integer/parseInt item))

(defn load-data
  [path]
  (->> path
       slurp
       cs/split-lines
       (map #(cs/split % #"\s"))
       flatten
       (map parse-int)))

(defn load-all-data
  []
  (into {}
        (for [idx (range 1 65)]
          [idx (load-data (format "resources/data%d.txt" idx))])))

(defn load-t
  [path]
  (->> path
       slurp
       cs/split-lines
       rest
       (map #(->> (cs/split % #"\s*,\s*")
                  (map parse-int)))
       (map #(vector (first %) (vec (rest %))))
       (into {})))

(defn resolve-results
  [dataset network]
  (into {}
        (for [entry dataset]
          [(key entry) (n/resolve-results network (val entry))])))

(defn resolve-deltas
  [results t-all network]
  (into {} (for [idx (keys results)]
             [idx (n/update-delta (results idx)
                                  (t-all idx)
                                  network)])))

(defn analyze-all
  [deltas]
  (into {} (for [idx (keys deltas)]
             [idx (n/differential (deltas idx))])))

(defn total-cost
  [deltas t-all]
  (reduce + (for [idx (range 1 65)]
              (n/cost (t-all idx) (deltas idx)))))

(defn learn
  [dataset t-set network eta d]
  (loop [network network]
    (let [delta-set (-> dataset
                        (resolve-results network)
                        (resolve-deltas t-set network))
          w (total-cost delta-set t-set)]
      (println w)
      (if (< w d)
        network
        (recur (n/train network eta (vals delta-set)))))))