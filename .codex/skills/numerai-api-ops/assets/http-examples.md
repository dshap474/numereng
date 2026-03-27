# GraphQL HTTP Examples (API-Only)

Endpoint:
- `https://api-tournament.numer.ai`

Set auth token for protected operations:

```bash
export NUMERAI_API_AUTH="Token PUBLIC_KEY$PRIVATE_KEY"
```

## Read-Only Examples

### List tournaments

```bash
curl -s https://api-tournament.numer.ai \
  -H 'Content-Type: application/json' \
  --data '{
    "query": "query{ tournaments { id name tournament active } }"
  }'
```

### Round details

```bash
curl -s https://api-tournament.numer.ai \
  -H 'Content-Type: application/json' \
  --data '{
    "query": "query($roundNumber:Int!,$tournament:Int!){ roundDetails(roundNumber:$roundNumber,tournament:$tournament){ roundNumber tournament status openTime closeTime scoreTime roundResolveTime totalSubmitted totalStakes } }",
    "variables": {"roundNumber": 1, "tournament": 8}
  }'
```

### Current round (Classic)

```bash
curl -s https://api-tournament.numer.ai \
  -H 'Content-Type: application/json' \
  --data '{
    "query": "query($tournament:Int!){ rounds(tournament:$tournament, number:0){ number openTime resolveTime resolvedGeneral resolvedStaking } }",
    "variables": {"tournament": 8}
  }'
```

### Token info and scopes (auth required)

```bash
curl -s https://api-tournament.numer.ai \
  -H 'Content-Type: application/json' \
  -H "Authorization: $NUMERAI_API_AUTH" \
  --data '{"query":"query{ apiTokenInfo { accountUsername name publicId scopes } apiTokenScopes { name description } }"}'
```

### List compute pickles (auth required)

```bash
curl -s https://api-tournament.numer.ai \
  -H 'Content-Type: application/json' \
  -H "Authorization: $NUMERAI_API_AUTH" \
  --data '{
    "query":"query($modelId:ID,$unassigned:Boolean!){ computePickles(modelId:$modelId, unassigned:$unassigned){ id filename validationStatus diagnosticsStatus triggerStatus assignedModelSlots insertedAt updatedAt } }",
    "variables":{"modelId":"<MODEL_ID>","unassigned":false}
  }'
```

## Write Flow Examples (confirm with user first)

### Create model

```bash
curl -s https://api-tournament.numer.ai \
  -H 'Content-Type: application/json' \
  -H "Authorization: $NUMERAI_API_AUTH" \
  --data '{
    "query":"mutation($name:String!,$tournament:Int!){ addModel(name:$name,tournament:$tournament){ id name tournament } }",
    "variables":{"name":"example-model","tournament":8}
  }'
```

### Classic/Crypto submission flow

1) Get upload auth URL:

```bash
curl -s https://api-tournament.numer.ai \
  -H 'Content-Type: application/json' \
  -H "Authorization: $NUMERAI_API_AUTH" \
  --data '{
    "query":"query($filename:String!,$tournament:Int!,$modelId:ID){ submissionUploadAuth(filename:$filename,tournament:$tournament,modelId:$modelId){ filename url accelerated } }",
    "variables":{"filename":"predictions.csv","tournament":8,"modelId":"<MODEL_ID>"}
  }'
```

2) PUT the file bytes to returned `url` (presigned storage URL).

3) Create submission record:

```bash
curl -s https://api-tournament.numer.ai \
  -H 'Content-Type: application/json' \
  -H "Authorization: $NUMERAI_API_AUTH" \
  --data '{
    "query":"mutation($filename:String!,$tournament:Int!,$modelId:ID,$triggerId:ID,$dataDatestamp:Int){ createSubmission(filename:$filename,tournament:$tournament,modelId:$modelId,triggerId:$triggerId,source:\"manual\",dataDatestamp:$dataDatestamp){ id filename insertedAt triggerId } }",
    "variables":{"filename":"<UPLOADED_FILENAME>","tournament":8,"modelId":"<MODEL_ID>","triggerId":null,"dataDatestamp":null}
  }'
```

### Signals submission flow

```bash
curl -s https://api-tournament.numer.ai \
  -H 'Content-Type: application/json' \
  -H "Authorization: $NUMERAI_API_AUTH" \
  --data '{
    "query":"query($filename:String!,$tournament:Int,$modelId:ID){ submissionUploadSignalsAuth(filename:$filename,tournament:$tournament,modelId:$modelId){ filename url accelerated } }",
    "variables":{"filename":"signals.csv","tournament":11,"modelId":"<MODEL_ID>"}
  }'
```

```bash
curl -s https://api-tournament.numer.ai \
  -H 'Content-Type: application/json' \
  -H "Authorization: $NUMERAI_API_AUTH" \
  --data '{
    "query":"mutation($filename:String!,$tournament:Int,$modelId:ID,$triggerId:ID,$dataDatestamp:Int){ createSignalsSubmission(filename:$filename,tournament:$tournament,modelId:$modelId,triggerId:$triggerId,source:\"manual\",dataDatestamp:$dataDatestamp){ id filename insertedAt triggerId } }",
    "variables":{"filename":"<UPLOADED_FILENAME>","modelId":"<MODEL_ID>"}
  }'
```

### Diagnostics flow

```bash
curl -s https://api-tournament.numer.ai \
  -H 'Content-Type: application/json' \
  -H "Authorization: $NUMERAI_API_AUTH" \
  --data '{
    "query":"query($filename:String!,$tournament:Int!,$modelId:String){ diagnosticsUploadAuth(filename:$filename,tournament:$tournament,modelId:$modelId){ filename url } }",
    "variables":{"filename":"predictions.csv","tournament":8,"modelId":"<MODEL_ID>"}
  }'
```

```bash
curl -s https://api-tournament.numer.ai \
  -H 'Content-Type: application/json' \
  -H "Authorization: $NUMERAI_API_AUTH" \
  --data '{
    "query":"mutation($filename:String!,$tournament:Int!,$modelId:String){ createDiagnostics(filename:$filename,tournament:$tournament,modelId:$modelId){ id } }",
    "variables":{"filename":"<UPLOADED_FILENAME>","tournament":8,"modelId":"<MODEL_ID>"}
  }'
```

### Delete diagnostics

```bash
curl -s https://api-tournament.numer.ai \
  -H 'Content-Type: application/json' \
  -H "Authorization: $NUMERAI_API_AUTH" \
  --data '{
    "query":"mutation($ids:[String!]!){ deleteDiagnostics(v2DiagnosticsIds:$ids) }",
    "variables":{"ids":["<DIAGNOSTICS_ID>"]}
  }'
```

### Assign and trigger compute pickle

```bash
curl -s https://api-tournament.numer.ai \
  -H 'Content-Type: application/json' \
  -H "Authorization: $NUMERAI_API_AUTH" \
  --data '{
    "query":"mutation($modelId:String,$pickleId:ID){ assignPickleToModel(modelId:$modelId,pickleId:$pickleId) }",
    "variables":{"modelId":"<MODEL_ID>","pickleId":"<PICKLE_ID>"}
  }'
```

```bash
curl -s https://api-tournament.numer.ai \
  -H 'Content-Type: application/json' \
  -H "Authorization: $NUMERAI_API_AUTH" \
  --data '{
    "query":"mutation($modelId:ID,$pickleId:ID,$triggerValidation:Boolean){ triggerComputePickleUpload(modelId:$modelId,pickleId:$pickleId,triggerValidation:$triggerValidation){ id triggerStatus updatedAt } }",
    "variables":{"modelId":"<MODEL_ID>","pickleId":"<PICKLE_ID>","triggerValidation":false}
  }'
```

### Stake change flow

```bash
curl -s https://api-tournament.numer.ai \
  -H 'Content-Type: application/json' \
  -H "Authorization: $NUMERAI_API_AUTH" \
  --data '{
    "query":"mutation($value:String!,$type:String!,$tournamentNumber:Int!,$modelId:String){ v2ChangeStake(value:$value,type:$type,tournamentNumber:$tournamentNumber,modelId:$modelId){ requestedAmount status type dueDate } }",
    "variables":{"value":"1.0","type":"increase","tournamentNumber":8,"modelId":"<MODEL_ID>"}
  }'
```

## Notes

- Prefer the templates under `assets/graphql-templates/` or `scripts/numerai_api_ops.py` instead of writing ad hoc requests.
- The live Classic submission roots are camelCase:
  - `submissionUploadAuth`
  - `createSubmission`
- For write operations, run read-back verification (`submissions`, `diagnostics`, `computePickles`, `model`, or profile queries) after mutation.
