(ns liu.mars.ml.neural
  (:require [clojure.string :as cs]
            [clojure.pprint :as pprint]))

;; 神经元定义为 {:w [权重列表] :b 偏置量}
;; 神经元计算结果为 {:z zeta :a alpha :d delta}
;; 正值为 t 的向量
(defn create-node
  "以随机的权重和偏移构造神经元单位"
  [prev-count]
  {:w (vec (repeatedly prev-count rand))
   :b (rand)})

(defn create-network
  [& layers]
  (assert (> (count layers) 2))
  (let [input (repeatedly (first layers) (fn [] identity))]
    (reduce (fn [result c]
              (let [prev (last result)
                    prev-count (count prev)]
                (->> (vec (repeatedly c #(create-node prev-count)))
                     (conj result))))
            [input]
            (rest layers))))

(defn sigmoid
  "sigma 函数 1/(1+e^-x)"
  [x]
  (/ 1
     (+ 1 (Math/exp (- x)))))

(defn d-sigmoid
  "sigmoid 函数的导数函数"
  [x]
  (* (sigmoid x) (- 1 (sigmoid x))))

(defn zeta
  "求解特定节点的多项式值，即 z，该函数仅针对隐藏层和输出层"
  [node prev-layer-results]
  (reduce + (:b node) (map #(* %1 (:a %2)) (:w node) prev-layer-results)))

(defn alpha
  "求解特定节点的输出值，即 a(z)，该函数仅针对隐藏层和输出层，这
  个值也可以在得到 zeta 的情况下调用点火函数得到，默认使用 sigmoid 作为点火函数"
  ([node prev-layer-values]
   (sigmoid (zeta node prev-layer-values)))
  ([node prev-layer-values fire]
   (fire (zeta node prev-layer-values))))

(defn output-delta
  "这里仅针对点火函数为 sigmoid 的输出层神经元给出 delta 值，接收的参数为 alpha值，zeta 和正值"
  [a z t]
  (* (- a t) (d-sigmoid z)))

(defn resolve-delta
  "利用误差反向传播法递推指定节点的误差 delta 值。参数为节点的 idx，计算结果，下一层节点和下一层的结果"
  [idx node-result next-layer next-layer-results]
  (let [weights (map #(-> % :w (nth idx)) next-layer)
        deltas (map :d next-layer-results)]
    (* (reduce + 0 (map * weights deltas))
       (d-sigmoid (:z node-result)))))

(defn resolve-results
  "根据给定的神经网络和输入值，给出一个结果向量，其内容是输入层的值和隐藏层到输出层的各层计算结果 {:a alpha :z zeta}"
  [network input]
  (let [first-layer (vec (map (fn [f p] {:a (f p)}) (first network) input))]
    (reduce (fn [results layer]
              (let [prev-layer (last results)
                    result (vec (for [node layer]
                                  (let [z (zeta node prev-layer)
                                        a (sigmoid z)]
                                    {:z z :a a})))]
                (conj results result))) [first-layer] (rest network))))

(defn update-hidden
  "依据已得到输出层 delta 的结果集，更新指定网络的隐藏层结果，将求得的 delta 附在结果集中"
  [result network]
  (loop [layer-index (- (count result) 2) result result]
    (if (= layer-index 0)
      result
      (recur
        (dec layer-index)
        (update
          result
          layer-index
          (fn [layer]
            (map-indexed
              (fn [index node]
                (assoc node
                  :d
                  (resolve-delta
                    index node
                    (nth network (inc layer-index))
                    (nth result (inc layer-index)))))
              layer)))))))

(defn update-delta
  "根据结果集和正值生成整个网络（隐藏层和输出层）的误差，生成与结果集合并的结果"
  [result t-vector network]
  (let [last-index (-> result count dec)]
    (-> result
        (update last-index (fn [layer] (map (fn [node t]
                                              (assoc node :d (output-delta (:a node) (:z node) t)))
                                            layer t-vector)))
        (update-hidden network))))

(defn node-differential
  "单节点微分函数，根据当前节点的计算结果（delta）以及上一层节点的计算结果（alpha），计算权重和偏置量的偏微分结果"
  [result prev-layer-results]
  {:w (vec (map #(* (:d result) (:a %)) prev-layer-results))
   :b (:d result)})

(defn layer-differential
  "层微分函数，根据当前层和上一层的计算结果，生成整层的微分结果。"
  [layer-results prev-layer-results]
  (vec (map #(node-differential % prev-layer-results) layer-results)))

(defn differential
  "根据 delta 结果集求得神经元网络的偏微分结果集"
  [result]
  (loop [layer-index 1 dataset [(first result)]]
    (if (= (-> result count) layer-index)
      dataset
      (let [next-layer (inc layer-index)]
        (recur next-layer
               (assoc dataset
                 layer-index
                 (layer-differential
                   (nth result layer-index)
                   (nth result (dec layer-index)))))))))

(defn create-offset
  "根据偏微分结果集和学习率构造出偏移量集合"
  [eta analyze-dataset]
  (let [merge-node (fn [x y]
                     {:w (vec (map + (:w x) (:w y)))
                      :b (+ (:b x) (:b y))})
        merge-layer (fn [x y]
                      (map merge-node x y))
        merge-couple (fn [x y]
                       (reduce conj
                               [(first x)]
                               (map merge-layer
                                    (rest x) (rest y))))
        update-node (fn [node]
                      (-> node
                          (update :w (fn [ws] (map #(* eta %) ws)))
                          (update :b * eta)))
        merged (reduce merge-couple analyze-dataset)]
    (reduce conj [(first merged)] (map #(vec (map update-node %)) (rest merged)))))

(defn node-walk
  "跟据给定的 offset 数据集生成修正后的 node"
  [node offset]
  (-> node
      (update :w #(vec (map - % (:w offset))))
      (update :b - (:b offset))))

(defn layer-walk
  "根据给定的 offset 层生成修订后的 layer"
  [layer offset]
  (map node-walk layer offset))

(defn train
  "根据 delta 结果集对神经网络进行训练"
  [network eta deltas]
  (reduce conj
          [(first network)]
          (map layer-walk
               (rest network)
               (rest (create-offset eta (map differential deltas))))))

(defn cost
  "计算代价函数的值"
  [t result]
  (let [output (->> (last result)
                    (map :a))]
    (/ (reduce + (map #(Math/pow (- %1 %2) 2) t output))
       2)))

(defn binary-pair
  "二值化处理，将一对[0, 1)之间的浮点数处理为 0 或 1 组成的数对，较大的值转为 1 ，较小的转为 0"
  ([x y]
   (if (> x y)
     [1 0]
     [0 1]))
  ([[x y]]
   (binary-pair x y)))