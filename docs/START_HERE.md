# Start here - kickoff

1. Run `pip install -r requirements.txt`.
2. Run `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`.
3. Open `/docs` and `/dashboard`.
4. In a second terminal run `python simulator/fleet_simulator.py`.
5. Confirm that several robots appear in the dashboard.
6. Run `git status`.
7. Create a branch: `git switch -c feature/persistence`.
8. Decide who will work on persistence, ML, API/dashboard and testing/documentation.

Do not begin by changing the simulator. The simulator is provided infrastructure for the challenge.
