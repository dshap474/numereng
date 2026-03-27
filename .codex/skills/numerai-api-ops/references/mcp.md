check_api_credentials verifies the API token, returns the account identity/scopes, and lists available scopes.

create_model creates a new Numerai model slot in a tournament.

get_current_round returns the latest round number, status, and open/close timing.

get_leaderboard fetches the public leaderboard with ranking and performance fields.

get_model_performance returns recent round-by-round performance history for a specific model ID.

get_model_profile looks up a model by name and returns its public profile/performance data.

get_round_details fetches detailed metadata for a specific round, including payout/timeline info.

get_tournaments lists active Numerai tournaments and their IDs.

graphql_query runs arbitrary GraphQL against Numerai, mainly for anything not covered by the named tools.

list_datasets lists dataset files available for a tournament round.

run_diagnostics manages diagnostics runs for prediction files: get upload auth, create, list, get, and delete.

upload_model manages Numerai Compute pickle uploads: get upload auth, create, list, assign, trigger, list runtimes/data
versions, and fetch logs.
