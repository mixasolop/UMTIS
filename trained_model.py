import joblib

from scraper import build_model_text, clean_text, guess_role, guess_seniority


ROLE_MODEL = joblib.load("job_classifier.pkl")
SENIORITY_MODEL = joblib.load("job_seniority_classifier.pkl")

TITLE = "Software Engineering Intern, Warsaw, Poland (Summer 2026)"
DESCRIPTION = """
About the job
What is Box?
Box (NYSE BOX) is the leader in Intelligent Content Management. Our platform enables organizations to fuel collaboration, manage the entire content lifecycle, secure critical content, and transform business workflows with enterprise AI. We help companies thrive in the new AI-first era of business. Founded in 2005, Box simplifies work for leading global organizations, including JLL, Morgan Stanley, and Nationwide. Box is headquartered in Redwood City, CA, with offices across the United States, Europe, and Asia.
By joining Box, you will have the unique opportunity to continue driving our platform forward. Content powers how we work. Itâ€™s the billions of files and information flowing across teams, departments, and key business processes every single day contracts, invoices, employee records, financials, product specs, marketing assets, and more. Our mission is to bring intelligence to the world of content management and empower our customers to completely transform workflows across their organizations. With the combination of AI and enterprise content, the opportunity has never been greater to transform how the world works together and at Box you will be on the front lines of this massive shift.
Why Box needs you
As a Software Engineering Intern, you will be embedded as a full-time member of one of our engineering teams, working on a summer project that impacts our product roadmap. Paired with your manager and a mentor, you will build and ship code to be released to the public. This is your time to shine and be proud of your project - after all, you are the one building it!
What Youâ€™ll Do
Build and complete a coding project. Drive your own development project, ask questions, design solutions, own and present the results. With your project you can make a true impact, whether it's external to our customers or internal to Boxers.
Be paired with a Software Engineer mentor. We'll pair you with a formal mentor on your team, but you can find a mentor anywhere in the company.
Attend team meetings. We treat you like a full-time engineer. You'll be immersed on your team and part of the daily stand-ups and other agile practices, getting an opportunity to really see how decisions get made.
Participate in intern events and programming. Through workshops and information sessions, we provide opportunities to meet other interns, grow your professional development skills, and expand your Box knowledge.
Write and review quality, testable, maintainable, and well-documented code.
Represent Box Poland internally and externally.
Who You Are
A 3rd year Computer Science/IT-related undergraduate student.
Currently based in Warsaw.
Proactively assess, communicate, and complete project milestones with manager and team members in a time sensitive manner.
You are passionate about solving complex problems using data-driven solutions.
Must speak English proficiently.
Preferred Skills
Knowledge of areas of the tech stack. If selected to be an intern for the summer, we will match you up with a team based on your interests and skillsets. We have teams ranging all over the tech stack and you can choose to explore front-end, back-end, or full stack opportunities, regardless of your prior experience.
Eagerness to learn additional coding languages. We like to use JavaScript, Java, PHP, Python, and Scala, and some other languages too.
Experience in a fast-paced, highly collaborative environment.
Experience working with distributed teams in different time zones. Weâ€™re a global company and our engineering teams span all the way from Poland to the US.
Percentage of Time Spent
60% coding / pair programming (specific to intern project work)
25% internal meetings (stand ups, weekly architecture & planning meetings, AMAâ€™s, 1 1â€™s)
15% intern community events (virtual or in-person workshops & events)
Tools and methodologies we use
Excited to learn new tools and methodologies such as Agile management - Scrum, Jira/Atlassian, GitHub Enterprise, Confluence, StackOverflow, GIT, and code reviews.
Pay 57 PLN per hour.
Applications will be considered on a rolling basis until the internship is filled. The internship will be 12-weeks long, working full-time hours over the course of summer 2026 (July - September).
If you're intrigued about this opportunity, but not sure you meet all the requirements, apply anyway.
"""


def show_prediction(model, title, description, label_name, fallback_label):
    text = build_model_text(title, clean_text(description))
    probabilities = model.predict_proba([text])[0]
    ranked = sorted(zip(model.classes_, probabilities), key=lambda item: item[1], reverse=True)
    prediction = ranked[0][0]

    if fallback_label and fallback_label != "Other" and ranked[0][1] < 0.40:
        prediction = fallback_label

    print(label_name)
    print(f"Prediction: {prediction}")
    print("Top classes:")
    for label, score in ranked[:3]:
        print(f"  {label}: {score:.3f}")
    print()


cleaned = clean_text(DESCRIPTION)
show_prediction(
    ROLE_MODEL,
    TITLE,
    DESCRIPTION,
    "Role Prediction",
    guess_role(TITLE, cleaned),
)
show_prediction(
    SENIORITY_MODEL,
    TITLE,
    DESCRIPTION,
    "Seniority Prediction",
    guess_seniority(TITLE, cleaned),
)
