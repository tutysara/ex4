
name := "ex4r"

version := "0.1"

organization := "org.tutysra"

scalaVersion := "2.9.1"

crossScalaVersions := Seq("2.8.2", "2.9.1")

resolvers ++= Seq(
  "ScalaNLP Maven2" at "http://repo.scalanlp.org/repo",
  "Scala Tools Snapshots" at "http://scala-tools.org/repo-snapshots/"
)

libraryDependencies ++= Seq(
  "com.googlecode.netlib-java" % "netlib-java" % "0.9.3",
  "jfree" % "jcommon" % "1.0.16",
  "jfree" % "jfreechart" % "1.0.13",
  "org.apache.xmlgraphics" % "xmlgraphics-commons" % "1.3.1", // for eps gen
  // "org.apache.xmlgraphics" % "batik-dom" % "1.7",    // for svg gen
  // "org.apache.xmlgraphics" % "batik-svggen" % "1.7", // for svg gen
  "com.lowagie" % "itext" % "2.1.5" intransitive(),  // for pdf gen
  "junit" % "junit" % "4.5" % "test",
  "org.scalala" %% "scalala" % "1.0.0.RC3-SNAPSHOT" ,//for scalala
  "org.scalanlp" %% "scalanlp-core" % "0.4.RC2-SNAPSHOT" //for scalanlp-core
)

libraryDependencies <<= (scalaVersion, libraryDependencies) { (sv, deps) =>
  deps :+ ("org.scala-lang" % "scala-compiler" % sv)
}

libraryDependencies <<= (scalaVersion, libraryDependencies) { (sv, deps) =>
  sv match {
    case "2.9.1" =>
      deps :+ ("jline" % "jline" % "0.9.94") // ("org.scala-lang" % "jline" % "2.9.1")
    case x if x.startsWith("2.8") =>
      deps :+ ("jline" % "jline" % "0.9.94")
    case x       => error("Unsupported Scala version " + x)
  }
}

libraryDependencies <<= (scalaVersion, libraryDependencies) { (sv, deps) =>
  sv match {
    case "2.9.1" =>
      (deps :+ ("org.scalatest" % "scalatest" % "1.4.RC2" % "test")
            :+ ("org.scala-tools.testing" %% "scalacheck" % "1.9" % "test"))
    case x if x.startsWith("2.8") =>
      (deps :+ ("org.scalatest" % "scalatest" % "1.3" % "test")
            :+ ("org.scala-tools.testing" % "scalacheck_2.8.1" % "1.8" % "test"))
    case x  => error("Unsupported Scala version " + x)
  }
}

publishTo <<= (version) { version: String =>
  val nexus = "http://nexus.scala-tools.org/content/repositories/"
  if (version.trim.endsWith("SNAPSHOT")) Some("snapshots" at nexus + "snapshots/") 
  else                                   Some("releases"  at nexus + "releases/")
}

credentials += Credentials(Path.userHome / ".ivy2" / ".credentials")

javacOptions ++= Seq("-source", "1.5", "-target", "1.5")

scalacOptions ++= Seq("-no-specialization","-deprecation","-target:jvm-1.5")

//initialCommands := scala.io.Source.fromFile("src/main/resources/scalala.scala").getLines.mkString("\n")

//mainClass in (Compile,packageBin) := Some("scalala.ScalalaConsole")

javaOptions += "-Xmx2g"

//seq(ProguardPlugin.proguardSettings :_*)


