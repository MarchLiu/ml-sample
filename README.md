# ml-samples

A Clojure library show a basic machine learn workflow.

## Usage

start into repl:
```shell script
lein repl
```

load dataset:

```clojure
(def dataset (load-all-data))
```
load t-values:
```clojure
(def t-all (load-t "resources/t.csv"))
```
create random network:
```clojure
(def network (n/create-network 12 3 2))
```
define a eta
```clojure
(def eta 0.2)
```
train and generate new network, you will see the cost being down to zero:
```clojure
(def new-network (learn dataset t-all network eta 0.1))
```
test new network:
```
(into (sorted-map)
      (map #(vector (key %) (->> % val last (map :a) n/binary-pair))
           results))
```

## License

Copyright © 2019 FIXME

This program and the accompanying materials are made available under the
terms of the Eclipse Public License 2.0 which is available at
http://www.eclipse.org/legal/epl-2.0.

This Source Code may also be made available under the following Secondary
Licenses when the conditions for such availability set forth in the Eclipse
Public License, v. 2.0 are satisfied: GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or (at your
option) any later version, with the GNU Classpath Exception which is available
at https://www.gnu.org/software/classpath/license.html.
