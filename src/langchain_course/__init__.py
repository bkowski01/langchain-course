from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


def main() -> None:
    load_dotenv()
    print("Hello from langchain-course!")
    information = """
    Jerry Lee Rice (born October 13, 1962)[7] is an American former professional football wide receiver who played for 20 seasons in the National Football League (NFL). He won three Super Bowl titles with the San Francisco 49ers before two shorter stints at the end of his career with the Oakland Raiders and Seattle Seahawks. For his accomplishments and numerous records, Rice is widely regarded as the greatest wide receiver of all time and one of the greatest players in NFL history.[8][9][10] His biography on the official Pro Football Hall of Fame website names him "the most prolific wide receiver in NFL history with staggering career totals".[11] In 1999, The Sporting News listed Rice second behind Jim Brown on its list of "Football's 100 Greatest Players".[12] In 2010, he was chosen by NFL Network's NFL Films production The Top 100: NFL's Greatest Players as the greatest player in NFL history.[13]

Rice played college football for four seasons with the Mississippi Valley State Delta Devils, setting several National Collegiate Athletic Association (NCAA) and team receiving records, including becoming the all-time leader in NCAA receiving touchdowns. He joined the 49ers after being selected with the 16th overall pick of the 1985 NFL draft. After a modest rookie season, Rice emerged in the following season as one of the best receivers in the league, leading the NFL in receiving yards and touchdowns, a feat he achieved six times. In 1987, Rice set the record for most receiving touchdowns in a season, with 22, in a twelve-game strike-shortened season. He won back-to-back championships in 1988 and 1989, and was the MVP of the former championship. Rice developed connections with quarterbacks Joe Montana and Steve Young that are viewed as among the best in NFL history, helping him lead the league in both receiving yards and touchdowns six times, and in receptions twice.[14]

Going into the 1990s, Rice won a third Super Bowl in 1994, and a second Offensive Player of the Year Award. After recovering from a knee injury and his play regressing, San Francisco released him in June 2001, where the Raiders would sign him to a four-year deal. He continued to start for the team, and helped lead them to an appearance in Super Bowl XXXVII, where they were defeated by the Tampa Bay Buccaneers, affecting Rice's previously unblemished Super Bowl record. Midway through 2004, the Raiders traded him to the Seahawks, where he spent his final season.[15][16] He briefly signed with the Broncos, retiring shortly before the start of the 2005 regular season.

Rice is the career leader in most major statistical categories for wide receivers,[17] including receptions, receiving touchdowns, receiving yards, scrimmage yards, and total touchdowns, holding the postseason records for these statistics, and once held the single-season records for yards and touchdowns. He scored more points than any other non-kicker in NFL history with 1,256.[18] Rice was selected to the Pro Bowl 13 times (1986–1996, 1998, 2002) and named All-Pro twelve times in his 20 NFL seasons, including ten First-team All-Pros, tied for the most by any player. Rice was inducted into the Pro Football Hall of Fame in 2010 and the College Football Hall of Fame in 2006. Rice was inducted into the Mississippi Sports Hall of Fame in 2007, and in the same year was inducted into the Bay Area Sports Hall of Fame. The NFL honored him as a member of the NFL 1980s All-Decade Team and the NFL 1990s All-Decade Team, as well as both the NFL 75th Anniversary All-Time Team and NFL 100th Anniversary All-Time Team.
"""

    summary_template = """
    Given the information {information} about a person, I want you to create:
    1. A short summary
    2. Two interesting facts about them
"""

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOllama(temperature=0, model="gemma4:e2b-mlx")
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)