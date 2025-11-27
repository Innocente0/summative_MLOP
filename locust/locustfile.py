from locust import HttpUser, task, between


class PredictUser(HttpUser):
    """
    Simulates a user sending prediction requests to the /predict endpoint.
    """

    wait_time = between(1, 3)  # seconds between tasks

    @task
    def predict_task(self):
        """
        Send a POST /predict with a sample image.
        Make sure 'locust/sample.jpg' exists in your project.
        """
        try:
            with open("locust/sample.jpg", "rb") as f:
                files = {"file": ("sample.jpg", f, "image/jpeg")}
                self.client.post("/predict", files=files)
        except FileNotFoundError:
            # Avoid crashing Locust if sample.jpg is missing
            pass
