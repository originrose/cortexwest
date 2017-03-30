(ns cortexwest.core
  (:require
    [clojure.java.browse :refer [browse-url]]
    [thi.ng.geom.viz.core :as viz]
    [thi.ng.geom.svg.core :as svg]
    [thi.ng.geom.core.vector :as t-vec]
    [thi.ng.color.core :as t-color]
    [thi.ng.math.core :as t-math :refer [PI TWO_PI]]
    [clojure.core.matrix :as mat]
    [clojure.core.matrix.stats :as stats]
    [cortex.nn.network :as network]
    [cortex.graph :as graph]
    [cortex.loss :refer [softmax-result-to-unit-vector]]
    [cortex.nn.execute :as execute]
    [cortex.nn.layers :as layers]
    ;[tsne.core :as tsne]
    ))

(def random (java.util.Random.))

(defn rand-gaussian
  []
  (.nextGaussian random))

; linear regression with basic line

; linear regression of polynomial

; mlp regression

; logistic regression classifier on some clusters

; mlp classifier on some clusters

(defn uuid
  []
  (str (java.util.UUID/randomUUID)))

(defn tmp-path
  []
  (str (System/getProperty "java.io.tmpdir")
       (uuid)
       ".svg"))

(defn export-viz
  [spec path]
  (->> spec
       (viz/svg-plot2d-cartesian)
       (svg/svg {:width 600 :height 600})
       (svg/serialize)
       (spit path)))

(defn show-viz
  [spec]
  (let [path (tmp-path)]
    (export-viz spec path)
    (browse-url (str "file://" path))))

(defn eq1
  [t]
  (let [x (t-math/mix (- PI) PI t)]
    [x (* (Math/cos (* 0.5 x)) (Math/sin (* x x x)))]))

(defn eq2
  [t]
  (let [x (t-math/mix (- PI) PI t)]
    [x (Math/pow (* (Math/cos (* 0.5 x)) (Math/sin (* x x x))) 2)]))

(defn poly3
  [a b c x]
  (+ (* a x) (* b x x) (* c x x x)))

(defn round10
  [v]
  (Math/pow 10
            (Math/round (+ 0.5
                           (Math/log10 v)
                           (- (Math/log10 5.5))))))
(defn major-axis-for
  [v]
  (let [axis (round10 v)
        n-ticks (/ v axis)]
    (cond
      (> n-ticks 12.0) (* 10 axis)
      (< n-ticks 3.0) (* 0.1 axis)
      :default axis)))

(defn viz-for
  [data]
  (let [margin-size 0.05
        xs (map first data)
        ys (map second data)
        min-x (apply min xs)
        max-x (apply max xs)
        xr (- max-x min-x)
        x-major (major-axis-for xr)
        x-margin (* margin-size xr)
        min-y (apply min ys)
        max-y (apply max ys)
        yr (- max-y min-y)
        y-margin (* margin-size yr)
        y-major (major-axis-for yr)]
    {:x-axis (viz/linear-axis
               {:domain [(- min-x x-margin) (+ max-x x-margin)]
                :range  [50 580]
                :major  x-major
                :minor  (* 0.5 x-major)
                :pos    250})
     :y-axis (viz/linear-axis
               {:domain      [(- min-y y-margin) (+ max-y y-margin)]
                :range       [250 20]
                :major       y-major
                :minor       (* 0.5 y-major)
                :pos         50
                :label-dist  15
                :label-style {:text-anchor "end"}})
     :grid   {:attribs {:stroke "#caa"}
              :minor-y true}
     :data []}))

(defn hsv-rainbow
  [& {:keys [n] :or {n 8}}]
  (map (fn [v]
         @(t-color/as-css (t-color/hsva v 0.9 0.9)))
       (range 0 1.0 (/ 1.0 n))))

(defn viz-add-data
  [viz data & {:keys [layout
                      color]}]
  (let [color (or color (rand-nth (hsv-rainbow)))
        layout (case layout
                 :line viz/svg-line-plot
                 :scatter viz/svg-scatter-plot
                 viz/svg-line-plot)]
    (update-in viz [:data] conj
               {:values  data
                :attribs {:fill "none" :stroke color}
                :layout  layout})))

(defn noisy-line
  [m b noise x]
  (+ (* m x) b (* noise (rand-gaussian))))

(defn regression
  []
  (let [x-data (range 0 10 0.1)
        y-data (map (partial noisy-line 0.3 0.5 0.5) x-data)
        data (map (fn [x y] {:x x :y y}) x-data y-data)
        network (network/linear-network
                  [(layers/input 1 1 1 :id :x)
                   (layers/linear 1 :id :y)])
        epoch-count 8
        network
        (loop [network network
               epoch 0]
         (if (> epoch-count epoch)
            (recur (execute/train network data :batch-size 3) (inc epoch))
            network))
        inferences (execute/run network (map #(dissoc % :y) data) :batch-size 50)
        model-data (map vector x-data (map (comp first :y) inferences))
        sample-data (map vector x-data y-data)
        viz (-> (viz-for sample-data)
                (viz-add-data sample-data :layout :scatter)
                (viz-add-data model-data :layout :line))]
    (show-viz viz)))


(defn evaluate-network
  [network test-ds & {:keys [label-key batch-size ]
                      :or {label-key :y
                           batch-size 10}}]
  (/ (Math/sqrt (->> (execute/run network test-ds :batch-size batch-size)
                     (map
                       (fn [a b]
                         (Math/pow (- (get a label-key)
                                      (first (get b label-key)))
                                   2))
                       test-ds)
                     (reduce +)))
     (count test-ds)))

(defn mlp-regression
  [& [net]]
  (let [data (map eq1 (t-math/norm-range 500))
        data (filter #(and (> (first %) 0)
                           (< (first %) 1.5))
                     data)
        x-data (map first data)
        y-data (mat/scale (mapv second data) 5)
        net-data (map (fn [x y] {:x x :y y}) x-data y-data)
        ;_ (println (take 50 net-data))
        test-ds (map #(dissoc % :y) net-data)
        network (network/linear-network
                  [(layers/input 1 1 1 :id :x)
                   (layers/batch-normalization)
                   (layers/linear->logistic 50)
                   (layers/linear->relu 30)
                   (layers/linear 1 :id :y)])
        epoch-count 8
        trained-network
        (loop [net network
               epoch 0]
         (if (> epoch-count epoch)
           (let [train-data (shuffle (apply concat (repeat 100 net-data)))
                 new-net (execute/train net train-data :batch-size 100)
                 mse (evaluate-network net net-data)]
             (println "mse: " mse)
            (recur new-net (inc epoch)))
            net))
        inferences (execute/run trained-network test-ds :batch-size 10)
        model-data (map vector x-data (map (comp first :y) inferences))
        sample-data (map vector x-data y-data)
        viz (-> (viz-for sample-data)
                (viz-add-data sample-data :layout :line)
                (viz-add-data model-data :layout :line))]
    (show-viz viz)))


(defn xy-cluster
  [x y n label]
  (repeatedly n
          (fn []
            {:input [(+ x (rand-gaussian))
                     (+ y (rand-gaussian))]
             :label label})))

(defn m->v
  [data]
  (map
    (fn [{:keys [input]}]
      [(first input) (second input)])
    data))


(defn classifier
  []
  (let [train-a (xy-cluster 30 50 50 [1 0])
        train-b (xy-cluster 20 10 50 [0 1])
        train-data (concat train-a train-b)
        test-a (xy-cluster 30 50 50 [1 0])
        test-b (xy-cluster 20 10 50 [0 1])
        test-data (concat test-a test-b)
        network (network/linear-network
                  [(layers/input 2 1 1 :id :input)
                   (layers/batch-normalization)
                   (layers/linear->relu 20)
                   (layers/linear 2)
                   (layers/softmax :output-channels 2 :id :label)])
        epoch-count 10
        network
        (loop [network network
               epoch 0]
         (if (> epoch-count epoch)
            (recur (execute/train network (apply concat (repeat 50 train-data)) :batch-size 10) (inc epoch))
            network))
        predictions (execute/run network (take 30 test-data) :batch-size 10)
        predictions (map #(update-in % [:label] softmax-result-to-unit-vector) predictions)
        predictions (map merge test-data predictions)
        model-data (group-by :label predictions)
        ;_ (println model-data)
        a-data (m->v (get model-data [1.0 0]))
        b-data (m->v (get model-data [0 1.0]))
        _ (println "results:" (count a-data) (count b-data))
        a-train-data (m->v train-a)
        b-train-data (m->v train-b)
        colors (hsv-rainbow)
        viz (-> (viz-for (concat a-train-data b-train-data))
                (viz-add-data a-train-data :layout :scatter :color (nth colors 0))
                (viz-add-data b-train-data :layout :scatter :color (nth colors 1))
                (viz-add-data a-data :layout :scatter :color (nth colors 5))
                (viz-add-data b-data :layout :scatter :color (nth colors 6))
                )]
    (show-viz viz)))


(defn plot-eq1
  []
  (let [data (map eq1 (t-math/norm-range 200))]
    (show-viz (viz-spec data :layout :line))))


