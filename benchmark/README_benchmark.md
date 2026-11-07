# Benchmark HomeLab V16 — 33 questions

Basé sur le corpus HomeLab RAG V16, snapshot du 03/11/2026, incluant les 27 documents de ingest_flat. Selon confirmation utilisateur, ce corpus est chargé dans Atlas. Les réponses évaluent ce snapshot documentaire, pas la réalité live des services. Les mises à jour canoniques priment sur les paragraphes historiques ou les métadonnées de statut devenus obsolètes.

## Utilisation

- Sélectionner le tenant réel de cette ingestion : ne pas présumer demo ou homelab.
- Copier la question seule dans le playground ; conserver les réponses attendues hors du corpus ingéré pour éviter de contaminer le test.
- Mapper les fichiers et document_id du front matter vers les vrais IDs persistés dans Atlas. Les champs expected_document_ids sont volontairement vides. Ne pas employer les IDs de Library comme IDs Atlas.
- JSON et CSV utilisent un format neutre : compatibilité d’import playground non vérifiée.
- Une liste de sources indique des références acceptables, pas nécessairement toutes les sources à retrouver. Pour une synthèse, contrôler la couverture de chaque fait requis.

## Comparaison

D’abord exécuter les 33 questions sans raisonnement avec vectoriel seul, hybride, vectoriel + reranking et hybride + reranking. Garder modèle, paramètres de génération, top-k et corpus identiques. Faire 3 à 5 répétitions, varier l’ordre et séparer le premier appel à froid. Comparer ensuite le raisonnement uniquement sur les synthèses ou erreurs persistantes.

Notation réponse : 0 = incorrecte/inventée, 1 = partielle, 2 = complète et correctement datée. Noter séparément validité des citations, priorité temporelle et abstention. Calculer Hit@5 après mapping des sources ; Recall@5 seulement avec un ensemble de pertinence annoté suffisamment complet. Mesurer embeddings, recherche, reranking, génération, TTFT et total séparément si disponibles. Conserver IDs de chunks et ordre avant/après reranking puis ordre du contexte LLM pour diagnostiquer le classement hybride.

## Isolation tenant — tests distincts

1. Répéter HL-01 avec le tenant du corpus, puis un tenant de test dont l’absence de documents a été vérifiée. Aucun chunk HomeLab ne doit apparaître pour ce dernier ; une erreur explicite est acceptable.
2. Répéter avec un second tenant existant dont les documents attendus sont connus et vérifier chaque tenant_id retourné, y compris citations et contexte final.
3. Vérifier séparément le tenant absent et une tentative d’override non autorisée avec un compte de test restreint. Résultat conforme au contrat d’authentification documenté ; aucun fallback silencieux vers un autre tenant.
4. Rejouer pour les quatre configurations de recherche. Une réponse correcte pour le tenant autorisé ne constitue pas un test d’isolation. Si aucun corpus de contrôle n’existe, marquer ces tests non exécutés plutôt que les déclarer réussis.

## Client CLI autonome

Le client `run_benchmark.py` appelle la Gateway via HTTP ; il ne dépend ni de ses
modules Python, ni d'un accès direct à Atlas/Ollama. Python 3.11 ou plus récent est
requis. Il fonctionne sous Linux et Windows. Depuis la racine du dépôt :

```sh
python -m pip install -r benchmark/requirements.txt
python benchmark/run_benchmark.py --help
```

Exécution d'une condition, avec le tenant réellement ingéré :

```sh
python benchmark/run_benchmark.py --base-url https://rag.home.thinselin.net --tenant TENANT_REEL --name vector --no-hybrid --no-rerank --top-k 5 --min-score 0.72 --num-candidates 100 --temperature 0 --reasoning none --max-tokens 2048 --top-p 1 --seed 42 --repetitions 3 --warmup 1 --shuffle --order-seed 42
```

La commande demande le bearer token sans l'afficher si `GATEWAY_API_KEY` n'est
pas fourni par l'environnement. Ne pas transmettre de secret en argument CLI ni
le placer dans un fichier de résultats. Pour une exécution non interactive,
injecter cette variable via le mécanisme sécurisé de votre terminal/ordonnanceur.
`--token-env AUTRE_VARIABLE` permet de choisir une autre variable. Une clé limitée
au tenant suffit ; une clé administrateur n'est nécessaire que pour
`--capture-settings`. Aucun fichier `.env` du serveur n'est chargé par le client.
HTTPS est vérifié ; `--ca-bundle benchmark/ca.pem` permet une autorité privée.
Les redirections ne sont pas suivies : utiliser directement l'URL canonique.

Le seul fichier de questions chargé par défaut est
`homelab_benchmark_v16_questions_only.json`. Le corrigé est rejeté comme entrée.
Les requêtes contiennent uniquement la question, sans réponse attendue, source
attendue, note, ID de question ni historique de conversation. Ne pas ingérer le
dossier benchmark dans le corpus RAG, particulièrement les corrigés/résultats.

### Matrice de comparaison

`matrix.example.json` fournit huit profils : vectoriel, hybride, vectoriel avec
reranking, hybride avec reranking, score minimum réduit, candidats augmentés,
raisonnement high et température augmentée. Les derniers profils modifient une
seule option par rapport au vectoriel de référence.

```sh
python benchmark/run_benchmark.py --tenant TENANT_REEL --matrix benchmark/matrix.example.json --max-tokens 2048 --top-p 1 --seed 42 --dry-run
python benchmark/run_benchmark.py --base-url https://rag.home.thinselin.net --tenant TENANT_REEL --matrix benchmark/matrix.example.json --max-tokens 2048 --top-p 1 --seed 42 --repetitions 3 --warmup 1 --shuffle --order-seed 42
```

Avec 33 questions, huit profils et trois répétitions, cela représente 792 appels
mesurés et huit warmups. Les sondes de retrieval optionnelles doublent les appels
pour les réponses chat réussies. Commencer par `--ids HL-01,HL-12 --repetitions 1`
ou `--limit 2`. Les profils sont exécutés séquentiellement dans l'ordre du JSON,
sans concurrence ni retry automatique. `--shuffle` mélange les questions de
chaque répétition avec un ordre reproductible et identique entre profils.
Inverser aussi l'ordre des profils lors d'une campagne supplémentaire pour
évaluer les effets de charge/cache. Un warmup ne garantit pas un état réellement
froid/chaud ; il est conservé, mais exclu des métriques mesurées.

Priorité des options : valeurs par défaut du client, objet JSON de
`--options-file`, options CLI communes, puis `options` du profil de matrice.
Les profils prennent donc le dessus sur les options CLI communes lorsqu'ils
définissent la même clé. Les clés inconnues sont rejetées, jamais silencieusement
acceptées. `--dry-run` écrit les plans, sans token ni appel réseau.

### Options réellement configurables à chaque requête

| Option HTTP | CLI / JSON | Forçage et limites |
|---|---|---|
| `hybrid_enabled` | `--hybrid` / `--no-hybrid` | Boolean explicite ; false force le vectoriel |
| `rerank_enabled` | `--rerank` / `--no-rerank` | Boolean explicite ; Atlas Native Reranking doit être disponible |
| `top_k` | `--top-k` | 1–50 ; nombre final de chunks, pas le top-k de sampling Ollama |
| `retrieval_min_score` | `--min-score` | 0–1 ; appliqué au score vectoriel, y compris après reranking ; ignoré en hybride |
| `retrieval_num_candidates` | `--num-candidates` | Entier positif ; Atlas utilise au moins la limite de résultats vectoriels |
| `temperature` | `--temperature` | 0–2 ; envoyée explicitement à Ollama |
| `reasoning_effort` | `--reasoning` | `none` pour demander sans raisonnement, `high` pour demander high ; support dépendant du modèle |
| `top_p`, `max_tokens`, `seed` | `--top-p`, `--max-tokens`, `--seed` | Transmis à Ollama ; fixer ces valeurs pour réduire les variables non contrôlées |
| `stop`, `presence_penalty`, `frequency_penalty`, `response_format` | Objet `--options-file` ou profil JSON | Options de génération transmises, support modèle à vérifier |

Le client force par défaut `hybrid_enabled=false`, `rerank_enabled=false`,
`top_k=5`, `retrieval_min_score=0.72`, `retrieval_num_candidates=100`,
`temperature=0` et `reasoning_effort=none`. Il ne laisse donc pas ces sept valeurs
dépendre implicitement des réglages du serveur. Les autres options de génération
non spécifiées restent au défaut du modèle et sont identifiées comme non forcées.
Une chaîne de raisonnement vide omet le champ côté Ollama : elle ne garantit pas
l'absence de raisonnement. Une réponse HTTP 200 ne prouve pas qu'un modèle a
honoré une option. Déployer d'abord la version Gateway supportant ces overrides ;
une ancienne version peut ignorer des champs d'un body non typé.

Le nom réel du modèle Ollama, le modèle/nombre de candidats de reranking, le nombre
de candidats hybrides, les index Atlas, le modèle d'embeddings, ses dimensions,
le corpus, le chunking et le timeout upstream sont des conditions serveur,
non surchargeables par ce client. Le champ chat `model` ne sélectionne pas le
modèle Ollama : la Gateway impose son modèle et retourne un alias public. Ne pas
envoyer `ollama_model` ou `retrieval_rerank_candidates` dans un profil : le client
les rejette. Modifier les réglages serveur, redémarrer/recréer la Gateway si
nécessaire, puis effectuer un nouveau run pour comparer ces conditions.
Les champs tools/tool_choice existent dans le forwarding Gateway, mais ne font
pas partie de ce benchmark de réponses sans outils.

En vectoriel pur, la limite intermédiaire est `top_k`. Avec reranking, elle vaut
`max(retrieval_rerank_candidates, top_k)` ; en hybride,
`max(retrieval_hybrid_candidates, top_k)` ; avec les deux, le minimum de ces deux
limites. Le nombre effectif de candidats est le maximum de cette limite et de
`retrieval_num_candidates`. Tester 20 contre 50 candidats avec une limite
intermédiaire de 100 ne change donc pas le nombre effectif. En hybride, `score`
est un score de reciprocal-rank fusion ; ne pas le comparer directement à une
similarité vectorielle. Le reranking précède la fusion sur la branche vectorielle.

### Conditions serveur et documents attendus

Pour conserver une configuration serveur non secrète :

```sh
python benchmark/run_benchmark.py --base-url https://rag.home.thinselin.net --tenant TENANT_REEL --name vector --capture-settings --repetitions 1
```

Cette option lit `GET /v1/admin/environment` avec la clé administrateur ; échec
explicite si elle n'est pas autorisée. Seule une liste blanche de champs non
secrets est conservée, jamais les clés, URI MongoDB ou registres d'authentification.
Alternative : `--server-settings benchmark/server_settings.json`, un objet avec
les noms des champs Settings, éventuellement sous une clé `settings`. Exemple :

```json
{
	"ollama_model": "qwen3.5:9b",
	"retrieval_rerank_model": "rerank-2.5",
	"retrieval_rerank_candidates": 100,
	"retrieval_hybrid_candidates": 100
}
```

Ces valeurs d'exemple doivent être remplacées par celles réellement configurées.
Le snapshot décrit la configuration, pas la preuve de son activation runtime.
L'endpoint admin relit les réglages ; des modifications en attente de restart
peuvent différer de l'état actif. Sans snapshot, les conditions serveur inconnues
restent null ; le client ne les invente pas. Vérifier séparément la stabilité du
corpus et la version réellement déployée.

`expected_document_ids` du corrigé est vide tant que les IDs Atlas ne sont pas
mappés. Ne pas considérer les IDs frontmatter du manifest comme des IDs Atlas.
`--document-map benchmark/document_id_mapping.json` ajoute une table indépendante
au résultat, sans la transmettre à la Gateway : objet dont chaque clé est un
nom de fichier ou ID frontmatter, et chaque valeur une liste d'IDs Atlas vérifiés.
Exemple de structure à renseigner avec les IDs observés :

```json
{"26-rag-gateway-validation.md": ["REMPLACER_PAR_ID_ATLAS_VERIFIE"]}
```

Les documents sont consultables via l'outil Chunks ou
`GET /v1/chunks/documents?tenant_id=TENANT_REEL`. Ne pas supposer qu'un filename
et un ID persistant sont identiques. Le juge peut comparer les réponses même
sans mapping, mais doit signaler une identité documentaire non vérifiée.

### Résultats, métriques et suivi live

Un JSON par profil est écrit sous `benchmark/results/`, nommé
`SESSION_PROFILE.json`. `--output-dir campaign01` choisit `benchmark/campaign01/` ;
un chemin en dehors de benchmark est rejeté. Les écritures sont atomiques avec un
checkpoint initial, après chaque réponse et après chaque sonde ; Ctrl+C marque
le run interrompu et conserve les réponses déjà obtenues. Pas de reprise
automatique : effectuer un nouveau run ou sélectionner les questions manquantes.

Le terminal affiche profil/question/répétition en cours, warmup ou mesure, puis
temps total, nombre de sources, tokens, compte des succès/erreurs et moyenne de
latence. Un heartbeat toutes les cinq secondes indique les requêtes encore en
attente ; `--progress-interval` le règle. Aucun bearer token n'est affiché. Le
secret d'authentification est également expurgé si le serveur le renvoie.
`--timeout` fixe le timeout HTTP client, mais ne modifie pas le timeout upstream
du serveur. Code de sortie : 0 = succès/plan, 1 = réponses en erreur, 2 = erreur
de configuration/initialisation, 130 = interruption.

Le fichier conserve l'intégralité du body JSON retourné (`raw_response`), les
payloads exacts sans headers secrets, l'ordre des questions, les options demandées,
les conditions serveur connues/inconnues, la version/hash du client, les erreurs,
le statut HTTP, les en-têtes de diagnostic autorisés, les réponses, traces de
raisonnement si présentes, finish_reason, usages tokens bruts et normalisés,
scores vectoriels/lexicaux/reranking/fusion, IDs de documents/chunks, métadonnées,
citations et étiquettes de citation invalides. Les champs inconnus du serveur sont
préservés dans le body brut. Une valeur absente est null, pas zéro.

`--capture-retrieval` effectue `POST /v1/retrieve` après chaque chat réussi, avec
les mêmes options de retrieval, pour obtenir le texte des chunks, leurs scores et
leur ordre. La sonde est conservée séparément avec son propre temps, et une
comparaison d'ordre document/chunk avec le chat. Elle n'est **jamais** présentée
comme le contexte exact du chat : deux appels peuvent produire des résultats
différents, même avec les mêmes options. Elle augmente les coûts et peut réchauffer
le cache pour les questions suivantes ; garder cette politique constante entre
les runs comparés. `--mode retrieve` permet un benchmark sans génération.

Limites de l'API actuelle : le chat ne retourne pas le texte exact du contexte
injecté ; les rangs avant reranking/fusion ne sont pas exposés. Le client utilise
des réponses non streamées afin de conserver l'usage complet retourné par Ollama.
TTFT, temps embeddings/recherche/reranking/génération séparés sont indisponibles
et restent null. D'éventuels timings upstream sont préservés tels quels, sans
assimiler une durée serveur à un stage non mesuré. Le temps client inclut réseau,
retrieval et génération, exclut les sondes séparées, mais inclut la charge runtime.
Les totaux tokens sont partiels si certaines réponses ne les rapportent pas.

### Analyse par un LLM juge

Joindre `JUDGE_PROMPT.md`, le corrigé `homelab_benchmark_v16.json` et un ou plusieurs
JSON de résultats terminés. Ajouter le mapping documentaire vérifié et le corpus
si une vérification fine du support des citations est nécessaire. Ne pas joindre
un plan dry-run comme une exécution mesurée. Le prompt demande une notation
0/1/2, l'analyse des sources retrouvées/citées, l'abstention, la chronologie, les
erreurs, les métriques et les comparaisons appariées par question/répétition.
Il sépare correction de la réponse et fidélité aux preuves réellement observées.
Conserver le rapport du juge et son bloc JSON d'analyse dans benchmark/results/.

### Tests locaux sans Gateway

```sh
python -m pip install pytest
python -m pytest benchmark/test_benchmark.py -q -p no:cacheprovider
```

Les tests utilisent des réponses HTTP simulées, sans accès au corpus Atlas ni
envoi d'un corrigé au modèle. Ils ne valident pas la disponibilité du déploiement
distant, son support des options de génération ni la qualité réelle des réponses.

## Questions et corrigé

### HL-01 — factuel

Quelle VM héberge la RAG Gateway dans le dernier état validé du corpus ?

Attendu : VM 106 docker, 192.168.2.21, déploiement validé le 30/10/2026.

Sources : 26-rag-gateway-validation.md

Contrôle : Rejeter NAS, xeon et migration encore à faire comme état final.

### HL-02 — factuel

Quelle est l’adresse IP du NAS nas01 ?

Attendu : 192.168.2.16.

Sources : 01-architecture.md

### HL-03 — lexical

Quels sous-réseau et passerelle sont associés au VLAN Guest 2 ?

Attendu : 192.168.5.0/24 ; passerelle 192.168.5.1.

Sources : 02-network-vlans.md

### HL-04 — lexical

Quel SSID est limité à la bande 2,4 GHz ?

Attendu : thinselinIoT.

Sources : 02-network-vlans.md

### HL-05 — lexical

Vers quelle adresse pointe nfsdocker et pourquoi son type DNS a-t-il changé ?

Attendu : Enregistrement A vers 192.168.2.16, pour corriger une boucle CNAME.

Sources : 03-dns-naming.md

### HL-06 — semantique

Quel service résout encore les anciens noms se terminant par thinselin.local ?

Attendu : AdGuard à 192.168.2.14, via délégation/forwarding du domaine historique.

Sources : 03-dns-naming.md

### HL-07 — factuel

Quel nom HTTPS permet d’accéder à la Gateway depuis le LAN ?

Attendu : https://rag.home.thinselin.net via Traefik.

Sources : 26-rag-gateway-validation.md

### HL-08 — lexical

Quels entrypoint et certresolver Traefik ont été validés pour la Gateway ?

Attendu : Entrypoint web-secure ; certresolver ovh.

Sources : 26-rag-gateway-validation.md

### HL-09 — lexical

Quel bind mount contient les données actives d’Open WebUI dans le relevé du 25 octobre ?

Attendu : /var/snap/docker/common/open-webui/data vers /app/backend/data.

Sources : 08-storage-nfs.md

### HL-10 — semantique

La base SQLite d’Open WebUI utilise-t-elle directement un montage NFS ?

Attendu : Non : bind mount sur ext4 dans la VM ; le disque virtuel de cette VM reste stocké sur NFS côté Proxmox.

Sources : 08-storage-nfs.md

### HL-11 — synthese

Si le NAS tombe, la Gateway hébergée sur la VM Docker reste-t-elle forcément utilisable ?

Attendu : Non : génération Ollama dépend du NAS et le disque de la VM dépend aussi du stockage NFS du NAS.

Sources : 08-storage-nfs.md, 25-rag-gateway-exploitation.md, 26-rag-gateway-validation.md

### HL-12 — chronologie

La Gateway est-elle encore seulement prévue sur la VM, ou son déploiement est-il validé ?

Attendu : Validé le 30/10/2026 ; les mentions de migration non confirmée sont supplantées par la mise à jour opérationnelle et le document 26.

Sources : 26-rag-gateway-validation.md

Contrôle : Tester les contradictions intra-document et inter-documents.

### HL-13 — chronologie

Quel GPU est documenté sur le NAS dans l’état le plus récent de ce corpus ?

Attendu : RTX 3060 12 Go/GiB selon V13 du 28/10 ; RTX 2060 6 Go correspond à un état historique. Revalidation live nécessaire pour l’état présent.

Sources : 10-ollama-gpu-tuning.md

### HL-14 — premisse_fausse

Sur quel port est raccordé le switch UniFi Flex 2.5G PoE installé dans le homelab ?

Attendu : La prémisse est fausse : le Flex a été envisagé mais n’est pas installé selon le corpus ; aucun port ne doit être inventé.

Sources : 01-architecture.md

### HL-15 — premisse_fausse

Le kill switch garantit-il que le Wi-Fi Guest ne sort jamais par Free ?

Attendu : Non : fuite vers le WAN lors de la désactivation manuelle du VPN ; échec d’authentification VPN bloque en revanche le trafic.

Sources : 04-vpn-guest-incident.md

### HL-16 — synthese

Pourquoi un test Guest avec de mauvais identifiants VPN ne suffit-il pas à valider le kill switch ?

Attendu : Le comportement diffère entre échec d’authentification et arrêt administratif : ce dernier laisse sortir via WAN. IPv6 et DNS doivent aussi être testés séparément.

Sources : 04-vpn-guest-incident.md

### HL-17 — semantique

Depuis Hermes sur le NAS, peut-on employer le même nom rag-gateway que depuis Open WebUI ?

Attendu : Non : alias du réseau Docker traefik-net de la VM ; Hermes doit passer par un endpoint joignable, notamment HTTPS Traefik.

Sources : 24-architecture-ia-rag.md, 26-rag-gateway-validation.md

### HL-18 — synthese

Quelles parties du RAG tournent sur la VM, sur le NAS et dans le cloud ?

Attendu : VM 106 : Gateway et Open WebUI/Traefik ; NAS : Ollama, génération GPU et Hermes ; cloud : Atlas chunks/recherche et AI Model APIs/Voyage pour embeddings.

Sources : 24-architecture-ia-rag.md, 26-rag-gateway-validation.md

### HL-19 — limite_preuve

La réponse citée dans le playground prouve-t-elle que les outils MCP Home Assistant fonctionnent via Hermes ?

Attendu : Non : le test RAG ne valide pas le raccordement Hermes ni les permissions et appels MCP Home Assistant.

Sources : 12-homeassistant-mcp.md, 26-rag-gateway-validation.md

### HL-20 — lexical

Quel outil a été demandé à Hermes pour consulter les pièces et étages Home Assistant ?

Attendu : ha_list_floors_areas ; essai observé, exécution finale non établie dans ce corpus.

Sources : 12-homeassistant-mcp.md

### HL-21 — diagnostic

Quelle correction est documentée pour stack.env not found dans Portainer ?

Attendu : Supprimer env_file: stack.env et transmettre les variables par environment: avec substitution depuis les variables de stack Portainer.

Sources : 26-rag-gateway-validation.md

### HL-22 — diagnostic

Pourquoi une clé Atlas AI Model APIs provoquait-elle un 401 lors des embeddings Voyage ?

Attendu : Clé Atlas utilisée contre api.voyageai.com ; employer le domaine Atlas AI Model APIs correspondant à la géographie de la clé, par exemple ai.mongodb.com pour une clé globale.

Sources : 26-rag-gateway-validation.md

### HL-23 — factuel

Combien de RK1 équipent la Turing Pi 2 et comment les rôles MongoDB sont-ils répartis ?

Attendu : 4 RK1 : 3 nœuds de données et 1 gestion/monitoring.

Sources : 13-rk1-cluster.md

### HL-24 — lexical

Quelle version du noyau personnalisé THP a été validée sur RK1 ?

Attendu : 5.10.160-rockchip-thp1.

Sources : 14-rk1-kernel-thp.md

### HL-25 — interpretation

La présence de madvise dans la liste sysfs prouve-t-elle que CONFIG_TRANSPARENT_HUGEPAGE_MADVISE est activé ?

Attendu : Non : symbole non activé dans le relevé ; défaut ALWAYS, sysfs [always] madvise never.

Sources : 14-rk1-kernel-thp.md

### HL-26 — interpretation

Quel gain de débit THP est documenté, et dans quelles conditions ?

Attendu : Petit dataset : environ 0,3 % d’écart ; grand dataset sous pression du cache WiredTiger : gain jusqu’à 7,1 % dans certains scénarios, pas un gain universel.

Sources : 15-mongodb-arm-bench.md

### HL-27 — abstention

Quel est le gain moyen consolidé de MongoDB 9 par rapport à MongoDB 8 sur les RK1 ?

Attendu : Impossible à déterminer : résultats consolidés et CSV bruts absents de la synthèse.

Sources : 15-mongodb-arm-bench.md

### HL-28 — abstention

Quelle est la température actuelle du GPU du NAS ?

Attendu : Impossible à connaître depuis ce corpus statique ; une mesure live est nécessaire. Ne pas recycler une température historique.

Sources : 00-start-here.md, 10-ollama-gpu-tuning.md

### HL-29 — abstention

Quelle est l’URL exacte authentifiée du serveur MCP Home Assistant et son jeton ?

Attendu : Endpoint exact non établi dans le corpus ; aucun jeton fourni et aucune valeur ne doit être inventée.

Sources : 12-homeassistant-mcp.md

### HL-30 — limite_preuve

La présence d’une allowlist Traefik suffit-elle à prouver que toutes les routes de la Gateway sont protégées ?

Attendu : Non : allowlist documentée pour LAN/WireGuard/Teleport, mais authentification par route et absence de chemins WAN doivent être vérifiées séparément.

Sources : 26-rag-gateway-validation.md

### HL-31 — multilingue

Where does the RAG Gateway run, and where is answer generation performed?

Attendu : Gateway on Docker VM 106, 192.168.2.21; answer generation through Ollama on NAS 192.168.2.16:30068.

Sources : 26-rag-gateway-validation.md

### HL-32 — multilingue

Which kernel release enabled Transparent Huge Pages on the RK1 nodes?

Attendu : 5.10.160-rockchip-thp1; avoid claiming deployment on every node is proven.

Sources : 14-rk1-kernel-thp.md



## Mise à jour V16 (2026-11-03)

Le manifeste inclut le document 27 sur le HomeLab AI Control Center. La question HL-33 couvre son hébergement sur la VM 106 et l’exposition limitée à l’interface privée/reverse proxy. Le corpus effectif dans Atlas et les IDs de documents persistés restent à vérifier avant d’interpréter les résultats. Les questions ne remplacent pas une validation de sécurité ni un test de trafic bout en bout.


### HL-33 — factuel

Sur quelle VM le HomeLab AI Control Center est-il hébergé, et comment son interface web est-elle exposée ?

Attendu : VM Docker 106 (`192.168.2.21`). Le frontend écoute sur `127.0.0.1:8080` et l’interface doit rester accessible par le réseau privé ou le reverse proxy, sans publication directe sur Internet. Les seules sondes de santé ne valident pas le trafic de bout en bout.

Sources : 27-homelab-ai-control-center.md

Contrôle : distinguer l’écoute locale de l’exposition réseau ; ne pas prétendre que toutes les fonctionnalités ou tous les flux sont validés.