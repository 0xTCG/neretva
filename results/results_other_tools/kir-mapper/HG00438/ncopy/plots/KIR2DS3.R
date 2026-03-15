
	options(warn=-1)

	if (!requireNamespace("ggplot2", quietly = TRUE)) {
	  
	  install.packages("ggplot2",
	                   repos = "http://cran.us.r-project.org")
	}

		if (!requireNamespace("plotly", quietly = TRUE)) {
	  
	  install.packages("plotly",
	                   repos = "http://cran.us.r-project.org")
	}
	
	if (!requireNamespace("htmlwidgets", quietly = TRUE)) {
	  
	  install.packages("htmlwidgets",
	                   repos = "http://cran.us.r-project.org")
	}
	
	if (!requireNamespace("stringr", quietly = TRUE)) {
	  
	  install.packages("stringr",
	                   repos = "http://cran.us.r-project.org")
	}
	
	
	library(ggplot2)
	library(plotly)
	library(htmlwidgets)
	library(stringr)

	setwd('results/HG00438//ncopy//plots/')
	db <- read.table('KIR2DS3_plot_db.txt',h=T,sep='\t')
	dbsub1 <- subset(db, Gene == 'KIR2DS3')
	dbsub2 <- subset(db, Gene == 'KIR2DS5')
	
	
	onecp = 0.280000 
	twocp = 0.700000 
	threecp = 1.300000
	fourcp = 1.700000


	count <- length(unique(db$Sample))

	p <- ggplot() + 
		geom_point(data=dbsub1, aes(x=Order,y=Ratio,group=Sample,color=Heterozygosis,shape=Gene),size=1) + 
		geom_point(data=dbsub2, aes(x=Order,y=Ratio,group=Sample,color=Heterozygosis,shape=Gene),size=1,alpha=0.5) + 
		theme_minimal() + 
		xlab('') +
		ylab('KIR2DS3 / KIR3DL3 ratio') +
		theme(
				axis.text.x = element_blank(),
				axis.ticks = element_blank(),
			legend.title = element_text(size = 8),
			legend.text = element_text(size = 6),
			plot.caption = element_text(size = 6)) +
		geom_hline(aes(yintercept=onecp, linetype = '0-1 copy threshold'),colour='red') + 
		geom_hline(aes(yintercept=twocp, linetype = '1-2 copy threshold'),colour='blue') + 
		geom_hline(aes(yintercept=threecp, linetype = '2-3 copy threshold'),colour='orange') + 
		geom_hline(aes(yintercept=fourcp, linetype = '3-4 copy threshold'),colour='green') + 
		scale_linetype_manual(name = 'Thresholds', values = c(2, 2, 2, 2), 
			guide = guide_legend(override.aes = list(color = c('red', 'blue', 'orange', 'green')))) +
		labs(title='KIR2DS3', caption = 'Calculated using the wgs mode')
	
	ggsave("KIR2DS3.png", width=8, height=6,bg = "white")
	
	myplot <- ggplotly(p)
	
	myplot <- ggplotly( p ) %>%
	  layout( legend=FALSE )
	
		for (i in 1:length(myplot$x$data)){
	  if (!is.null(myplot$x$data[[i]]$name)){
	    myplot$x$data[[i]]$name =  gsub("\\(","",str_split(myplot$x$data[[i]]$name,",")[[1]][1])
	  }
	}
	
	saveWidget(myplot, 'KIR2DS3.html', selfcontained = F, libdir = 'KIR2DS3')
	